use std::{sync::Arc, thread::JoinHandle};

use bitcask::{Bitcask, Config};
use rand::Rng;
use std::time::Duration;

pub fn main() -> std::io::Result<()> {
  let dir = tempfile::tempdir()?;
  let directory = dir.path().to_str().unwrap().to_owned();

  let bitcask = Arc::new(Bitcask::new(Config {
    directory,
    writer: true,
    max_active_file_size_in_bytes: 5,
    merge_files_after_n_files_created: 3,
  })?);

  let handles: Vec<JoinHandle<()>> = (0..8)
    .map(|_| {
      let bitcask = Arc::clone(&bitcask);
      std::thread::spawn(move || {
        let key = b"key";

        bitcask.delete(key).expect("error deleting key");

        std::thread::sleep(Duration::from_secs(rand::thread_rng().gen_range(1..=5)));

        println!("{:?} - {:?}", std::thread::current().id(), bitcask.get(key));

        bitcask.delete(key).expect("error deleting key");

        bitcask
          .put(key.to_vec(), b"value")
          .expect("error putting key");
      })
    })
    .collect();

  for handle in handles {
    handle.join().expect("error joining thread");
  }

  Ok(())
}
