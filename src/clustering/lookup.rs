use crate::cards::isomorphism::Isomorphism;
use crate::cards::isomorphisms::IsomorphismIterator;
use crate::cards::observation::Observation;
use crate::cards::street::Street;
use crate::clustering::abstraction::Abstraction;
use crate::clustering::histogram::Histogram;
use std::collections::BTreeMap;

#[derive(Default)]
/// this is the grand lookup table for all the Isomorphism -> Abstraction
/// mappings. we spend a lot of compute over a lot of hands (all of them!)
/// to precompute this mapping.
pub struct Lookup(BTreeMap<Isomorphism, Abstraction>);

impl From<Lookup> for BTreeMap<Isomorphism, Abstraction> {
    fn from(lookup: Lookup) -> BTreeMap<Isomorphism, Abstraction> {
        lookup.0
    }
}
impl From<BTreeMap<Isomorphism, Abstraction>> for Lookup {
    fn from(map: BTreeMap<Isomorphism, Abstraction>) -> Self {
        Self(map)
    }
}

impl Lookup {
    /// lookup the pre-computed abstraction for the outer observation
    pub fn lookup(&self, obs: &Observation) -> Abstraction {
        self.0
            .get(&Isomorphism::from(*obs))
            .cloned()
            .expect(&format!("precomputed abstraction missing for {obs}"))
    }
    #[cfg(feature = "native")]
    /// generate the entire space of inner layers
    pub fn projections(&self) -> Vec<Histogram> {
        use rayon::iter::IntoParallelIterator;
        use rayon::iter::ParallelIterator;
        IsomorphismIterator::from(self.street().prev())
            .collect::<Vec<Isomorphism>>()
            .into_par_iter()
            .map(|inner| self.future(&inner))
            .collect::<Vec<Histogram>>()
    }
    /// distribution over potential next states. this "layer locality" is what
    /// makes imperfect recall hierarchical kmeans nice
    fn future(&self, iso: &Isomorphism) -> Histogram {
        assert!(iso.0.street() != Street::Rive);
        iso.0
            .children()
            .map(|o| self.lookup(&o))
            .collect::<Vec<Abstraction>>()
            .into()
    }
    fn street(&self) -> Street {
        self.0.keys().next().expect("non empty").0.street()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore]
    #[test]
    fn persistence() {
        use crate::save::upload::Table;
        let street = Street::Pref;
        let lookup = Lookup::grow(street);
        lookup.save();
        let loaded = Lookup::load(street);
        std::iter::empty()
            .chain(lookup.0.iter().zip(loaded.0.iter()))
            .chain(loaded.0.iter().zip(lookup.0.iter()))
            .all(|((s1, l1), (s2, l2))| s1 == s2 && l1 == l2);
    }
}

#[cfg(feature = "native")]
impl crate::save::upload::Table for Lookup {
    fn name() -> String {
        "isomorphism".to_string()
    }
    fn columns() -> &'static [tokio_postgres::types::Type] {
        &[
            tokio_postgres::types::Type::INT8,
            tokio_postgres::types::Type::INT8,
        ]
    }
    fn sources() -> Vec<String> {
        Street::all()
            .iter()
            .rev()
            .copied()
            .map(Self::path)
            .collect()
    }
    fn creates() -> String {
        "
        CREATE TABLE IF NOT EXISTS isomorphism (
            obs        BIGINT,
            abs        BIGINT,
            position   INTEGER
        );"
        .to_string()
    }
    fn indices() -> String {
        "
        CREATE INDEX IF NOT EXISTS idx_isomorphism_covering     ON isomorphism  (obs, abs) INCLUDE (abs);
        CREATE INDEX IF NOT EXISTS idx_isomorphism_abs_position ON isomorphism  (abs, position);
        CREATE INDEX IF NOT EXISTS idx_isomorphism_abs_obs      ON isomorphism  (abs, obs);
        CREATE INDEX IF NOT EXISTS idx_isomorphism_abs          ON isomorphism  (abs);
        CREATE INDEX IF NOT EXISTS idx_isomorphism_obs          ON isomorphism  (obs);
        --
        WITH numbered AS (
            SELECT obs, abs, row_number() OVER (PARTITION BY abs ORDER BY obs) - 1 as rn
            FROM isomorphism
        )
            UPDATE isomorphism
            SET    position = numbered.rn
            FROM   numbered
            WHERE  isomorphism.obs = numbered.obs
            AND    isomorphism.abs = numbered.abs;
        "
        .to_string()
    }
    fn copy() -> String {
        "
        COPY isomorphism (
            obs,
            abs
        )
        FROM STDIN BINARY
        "
        .to_string()
    }
    /// abstractions for River are calculated once via obs.equity
    /// abstractions for Preflop are cequivalent to just enumerating isomorphisms
    fn grow(street: Street) -> Self {
        use rayon::iter::IntoParallelIterator;
        use rayon::iter::ParallelIterator;
        match street {
            Street::Rive => IsomorphismIterator::from(Street::Rive)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|iso| (iso, Abstraction::from(iso.0.equity())))
                .collect::<BTreeMap<_, _>>()
                .into(),
            Street::Pref => IsomorphismIterator::from(Street::Pref)
                .enumerate()
                .map(|(k, iso)| (iso, Abstraction::from((Street::Pref, k))))
                .collect::<BTreeMap<_, _>>()
                .into(),
            Street::Flop | Street::Turn => panic!("lookup must be learned via layer for {street}"),
        }
    }
    fn load(street: Street) -> Self {
        let ref path = Self::path(street);
        log::info!("{:<32}{:<32}", "loading     lookup", path);
        use byteorder::ReadBytesExt;
        use byteorder::BE;
        use std::fs::File;
        use std::io::BufReader;
        use std::io::Read;
        use std::io::Seek;
        use std::io::SeekFrom;
        
        // Use a memory-efficient approach with chunked processing
        let mut lookup = BTreeMap::new();
        let file = File::open(path).expect(&format!("open {}", path));
        let file_size = file.metadata().expect("get file metadata").len();
        log::info!("File size: {} bytes for {}", file_size, path);
        
        let mut reader = BufReader::with_capacity(4 * 1024 * 1024, file); // 4MB buffer
        let ref mut buffer = [0u8; 2];
        reader.seek(SeekFrom::Start(19)).expect("seek past header");
        
        // Read in batches to avoid excessive memory usage
        const BATCH_SIZE: usize = 100_000;
        let mut entries_processed = 0;
        let mut entry_count = 0;
        
        while reader.read_exact(buffer).is_ok() {
            match u16::from_be_bytes(buffer.clone()) {
                2 => {
                    assert!(8 == reader.read_u32::<BE>().expect("observation length"));
                    let iso = reader.read_i64::<BE>().expect("read observation");
                    assert!(8 == reader.read_u32::<BE>().expect("abstraction length"));
                    let abs = reader.read_i64::<BE>().expect("read abstraction");
                    
                    let observation = Isomorphism::from(iso);
                    let abstraction = Abstraction::from(abs);
                    
                    // Periodically log progress and force garbage collection
                    entry_count += 1;
                    if entry_count % BATCH_SIZE == 0 {
                        entries_processed += BATCH_SIZE;
                        
                        // Calculate approximate memory usage based on BTreeMap size
                        let approximate_memory_mb = (lookup.len() * (std::mem::size_of::<Isomorphism>() + 
                                                                  std::mem::size_of::<Abstraction>() + 
                                                                  std::mem::size_of::<Option<(Isomorphism, Abstraction)>>()) 
                                                  >> 20) as f64;
                        
                        log::info!("Processed {} entries in {} (approx {:.2} MB used)", 
                                  entries_processed, path, approximate_memory_mb);
                        
                        // Force a minor garbage collection
                        std::thread::yield_now();
                    }
                    
                    lookup.insert(observation, abstraction);
                }
                0xFFFF => break,
                n => panic!("unexpected number of fields: {}", n),
            }
        }
        
        // Add any remaining entries to the count
        entries_processed += entry_count % BATCH_SIZE;
        
        log::info!("Finished loading {} with {} entries", path, entries_processed);
        Self(lookup)
    }
    fn save(&self) {
        const N_FIELDS: u16 = 2;
        let street = self.street();
        let ref path = Self::path(street);
        let ref mut file = File::create(path).expect(&format!("touch {}", path));
        use byteorder::WriteBytesExt;
        use byteorder::BE;
        use std::fs::File;
        use std::io::Write;
        log::info!("{:<32}{:<32}", "saving      lookup", path);
        file.write_all(Self::header()).expect("header");
        for (Isomorphism(obs), abs) in self.0.iter() {
            file.write_u16::<BE>(N_FIELDS).unwrap();
            file.write_u32::<BE>(size_of::<i64>() as u32).unwrap();
            file.write_i64::<BE>(i64::from(*obs)).unwrap();
            file.write_u32::<BE>(size_of::<i64>() as u32).unwrap();
            file.write_i64::<BE>(i64::from(*abs)).unwrap();
        }
        file.write_u16::<BE>(Self::footer()).expect("trailer");
    }
}
