use std::{
  collections::HashMap,
  fs::{File, OpenOptions},
  io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
  path::Path,
  sync::RwLock,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc::crc32;
use tracing::{error, info, instrument};

#[derive(Debug, PartialEq, Clone)]
struct IndexEntry {
  /// Monotonically increasing id starting from 0.
  file_id: usize,
  /// The number of bytes that make up the value.
  value_len: u32,
  /// The position in the file identified by `file_id` where the
  /// record pointed by this entry starts.
  record_starts_at_position: u64,
  /// Timestamp of when the value was appended.
  timestamp: i64,
}

/// Represents a Bitcask record.
///
/// A Bitcask record looks like this on disk:
///                             ┌──────────size of───────────┐
///                             │                            │
/// ┌──────────┬───────────┬────┴─────┬────────────┬─────────▼─────────┬───────────────────────────────┐
/// │ checksum │ timestamp │  key_len │  value_len │        key        │             value             │
/// └──────────┴───────────┴──────────┴────────────┴───────────────────┴───────────────────────────────┘
///      u32        u32        u32          u32        [u8; key_len]             [u8; value_len]
///                                          │                                          ▲
///                                          │                                          │
///                                          └───────────────────────size of────────────┘
#[derive(Debug)]
struct Record {
  checksum: u32,
  timestamp: u32,
  key: Vec<u8>,
  value: Vec<u8>,
}

#[derive(Debug)]
struct MutableState {
  /// Maps a key to an entry containing information about where the
  /// data that belongs to the key is stored.
  index: HashMap<Vec<u8>, IndexEntry>,
  /// The size of the file that new entries will be written to.
  /// Used to know when we need to closet the file and create a new one.
  active_file_size_in_bytes: u64,
  /// The file that new entries will be written to.
  active_file: File,
  /// The id given to `active_file`.
  active_file_id: usize,
}

/// Value assigned to records that where the key has been deleted.
// TODO: this occupies some space, can we occupy less space?
static TOMBSTONE_VALUE: &[u8] = b"__bitcask__tombstone__";

#[derive(Debug)]
pub struct Bitcask {
  state: RwLock<MutableState>,
  config: Config,
}

#[derive(Debug)]
pub struct Config {
  /// Directory where Bitcask will store its files.
  pub directory: String,
  /// True if this instance of Bitcask is the one that handles writes.
  /// There should always be only once instance that writes to files.
  pub writer: bool,
  /// The file being written to will be closed when its size is surpasses the maximum file size
  /// and a new file will be created.
  pub max_active_file_size_in_bytes: u64,
}

impl Bitcask {
  fn get_file_id<P: AsRef<Path>>(directory: &P) -> std::io::Result<usize> {
    // Ensure `directory` exists.
    std::fs::create_dir_all(directory)?;

    let file_names: Vec<String> = std::fs::read_dir(directory)?
      .filter(|entry| entry.is_ok())
      .map(|entry| entry.unwrap().file_name())
      .map(|file_name| file_name.into_string().unwrap())
      // Ensure we only take files that were created by Bitcask.
      .filter(|file_name| file_name.ends_with(".bitcaskdata"))
      .collect();

    // Given a directory that has a bunch of Bitcask files, we want to find out which file is the latest
    // and we can do that by using the index that is added to each file.
    //
    // If we have the following files:
    //
    // dir/0.bitcaskdata
    // dir/1.bitcaskdata
    // dir/2.bitcaskdata
    //
    // we know that the latest file is file number 2.
    let latest_id = file_names
      .iter()
      .map(|file_name| file_name.split('.').collect::<Vec<_>>())
      .filter_map(|file_name_pieces| file_name_pieces.first().cloned())
      .map(|offset| offset.parse::<usize>().unwrap())
      .max();

    let new_id = match latest_id {
      None => 0,
      Some(id) => id + 1,
    };

    Ok(new_id)
  }

  #[instrument(name = "Creating Bitcask instance", skip_all, fields(config = ?config))]
  pub fn new(config: Config) -> std::io::Result<Self> {
    let directory: &Path = config.directory.as_ref();

    let file_id = Bitcask::get_file_id(&directory)?;

    let data_file_path = directory.join(format!("{}.bitcaskdata", file_id));

    let active_file = OpenOptions::new()
      .read(true)
      .create(true)
      .append(config.writer)
      .open(data_file_path)?;

    let active_file_size_in_bytes = active_file.metadata()?.len();

    let state = MutableState {
      active_file_id: file_id,
      index: HashMap::new(),
      active_file_size_in_bytes,
      active_file,
    };

    Ok(Self {
      state: RwLock::new(state),
      config,
    })
  }

  /// Appends a new record to the current active file.
  #[instrument(
    name = "put",
    skip_all,
    fields(
      key_utf8 = ?String::from_utf8_lossy(&key), key = ?key,
      value_utf8 = ?String::from_utf8_lossy(&value), value = ?value
    )
  )]
  pub fn put(&self, key: Vec<u8>, value: Vec<u8>) -> std::io::Result<()> {
    const CHECKSUM_SIZE_IN_BYTES: u64 = 4;
    const TIMESTAMP_SIZE_IN_BYTES: u64 = 4;
    const KEY_LEN_SIZE_IN_BYTES: u64 = 4;
    const VALUE_LEN_SIZE_IN_BYTES: u64 = 4;

    let key_len = key.len();
    let value_len = value.len();

    let entry_size_in_bytes = CHECKSUM_SIZE_IN_BYTES
      + TIMESTAMP_SIZE_IN_BYTES
      + KEY_LEN_SIZE_IN_BYTES
      + VALUE_LEN_SIZE_IN_BYTES
      + key_len as u64
      + value_len as u64;

    let mut content_for_checksum = Vec::with_capacity(key_len + value_len);

    content_for_checksum.extend_from_slice(&key);
    content_for_checksum.extend_from_slice(&value);

    let checksum = crc32::checksum_ieee(&content_for_checksum);

    let timestamp = chrono::Utc::now().timestamp();

    let mut state = self.state.write().unwrap();

    let record_starts_at_position;

    {
      let mut writer = BufWriter::new(&mut state.active_file);

      record_starts_at_position = writer.seek(SeekFrom::End(0))?;

      writer.write_u32::<LittleEndian>(checksum)?;
      writer.write_u32::<LittleEndian>(timestamp as u32)?;
      writer.write_u32::<LittleEndian>(key_len as u32)?;
      writer.write_u32::<LittleEndian>(value_len as u32)?;
      writer.write_all(&key)?;
      writer.write_all(&value)?;

      writer.flush()?;
    }

    let entry = IndexEntry {
      file_id: state.active_file_id,
      value_len: value_len as u32,
      timestamp,
      record_starts_at_position,
    };

    info!(?entry, "creating index entry");
    let _ = state.index.insert(key, entry);

    state.active_file_size_in_bytes += entry_size_in_bytes;

    Ok(())
  }

  /// Returns the value associated with the key.
  #[instrument(
    name = "get",
    skip_all,
    fields(key_utf8 = ?String::from_utf8_lossy(key), key = ?key)
  )]
  pub fn get(&self, key: &[u8]) -> std::io::Result<Option<Vec<u8>>> {
    let entry = {
      let state = self.state.read().unwrap();
      match state.index.get(key) {
        None => return Ok(None),
        // Cloning this should be fine since
        // IndexEntry size is 4 * 64 bits (i guess).
        Some(entry) => entry.clone(),
      }
    };

    info!(?entry, "entry file id found");

    let path: &Path = self.config.directory.as_ref();
    let file_path = path.join(format!("{}.bitcaskdata", entry.file_id));

    info!(?file_path, "opening bitcaskdata file");

    let reader = match File::open(&file_path) {
      Err(e) => {
        error!(error = ?e, "bitcaskdata file not found");
        return Err(e);
      }
      Ok(file) => BufReader::new(file),
    };

    let record = Bitcask::process_record(reader, entry.record_starts_at_position)?;

    info!(?record, "record found");

    Ok(Some(record.value))
  }

  // Reads a record into memory.
  #[instrument(name = "Processing record", skip_all, fields(record_starts_at_position = record_starts_at_position))]
  fn process_record(
    mut reader: impl Read + Seek,
    record_starts_at_position: u64,
  ) -> std::io::Result<Record> {
    // Move the reader to where the record starts.
    reader.seek(SeekFrom::Start(record_starts_at_position))?;

    let expected_checksum = reader.read_u32::<LittleEndian>()?;
    info!(checksum = expected_checksum, "read checksum from file");

    let timestamp = reader.read_u32::<LittleEndian>()?;
    info!(timestamp, "read timestamp from file");

    let key_len = reader.read_u32::<LittleEndian>()?;
    info!(key_len, "read key length from file");

    let value_len = reader.read_u32::<LittleEndian>()?;
    info!(value_len, "read value length from file");

    let mut buffer = Vec::with_capacity((key_len + value_len) as usize);

    reader
      .take((key_len + value_len) as u64)
      .read_to_end(&mut buffer)?;

    let checksum = crc32::checksum_ieee(&buffer);

    if expected_checksum != checksum {
      panic!(
        "data corruption found. expected_checksum={} actual_checksum={}",
        expected_checksum, checksum
      );
    }

    let value = buffer.split_off(key_len as usize);

    let key = buffer;

    Ok(Record {
      checksum,
      timestamp,
      key,
      value,
    })
  }

  /// Forgets about the key.
  #[instrument(
    name = "delete",
    skip_all,
    fields(key_utf8 = ?String::from_utf8_lossy(key), key = ?key)
  )]
  pub fn delete(&self, key: &[u8]) -> std::io::Result<()> {
    let key_len = key.len();
    let value_len = TOMBSTONE_VALUE.len();

    let mut content_for_checksum = Vec::with_capacity(key_len + value_len);

    content_for_checksum.extend_from_slice(key);
    content_for_checksum.extend_from_slice(TOMBSTONE_VALUE);

    let checksum = crc32::checksum_ieee(&content_for_checksum);

    let timestamp = chrono::Utc::now().timestamp();

    let mut state = self.state.write().unwrap();

    let record_starts_at_position;

    {
      let mut writer = BufWriter::new(&mut state.active_file);

      record_starts_at_position = writer.seek(SeekFrom::End(0))?;

      writer.write_u32::<LittleEndian>(checksum)?;
      writer.write_u32::<LittleEndian>(timestamp as u32)?;
      writer.write_u32::<LittleEndian>(key_len as u32)?;
      writer.write_u32::<LittleEndian>(value_len as u32)?;
      writer.write_all(key)?;
      writer.write_all(TOMBSTONE_VALUE)?;

      writer.flush()?;
    }

    let entry = IndexEntry {
      file_id: state.active_file_id,
      value_len: value_len as u32,
      timestamp,
      record_starts_at_position,
    };

    info!(?entry, "creating index entry");
    let _ = state.index.remove(key);

    Ok(())
  }

  /// Returns a list of the keys we have at the moment.
  #[instrument(name = "list_keys", skip_all)]
  pub fn list_keys(&self) -> Vec<Vec<u8>> {
    let state = self.state.read().unwrap();

    let keys: Vec<Vec<u8>> = state.index.keys().cloned().collect();

    info!(num_keys = keys.len());

    keys
  }
}

#[cfg(test)]
mod tests {
  use std::collections::HashSet;

  use tempfile::TempDir;

  use super::*;

  fn path(dir: &TempDir) -> String {
    dir.path().to_str().unwrap().to_owned()
  }

  fn bytes(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
  }

  fn default_bitcask(directory: String) -> std::io::Result<Bitcask> {
    Bitcask::new(Config {
      directory,
      writer: true,
      max_active_file_size_in_bytes: 1024 * 1024,
    })
  }

  #[test_log::test]
  fn each_bitcask_instance_creates_a_new_data_file() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let directory = path(&temp_dir);

    for i in 0..3 {
      let bitcask = default_bitcask(directory.clone())?;
      assert_eq!(i, bitcask.state.read().unwrap().active_file_id);
    }

    // [
    //  "/tmp/.tmpWluSFr/0.bitcaskdata",
    //  "/tmp/.tmpWluSFr/1.bitcaskdata",
    //  "/tmp/.tmpWluSFr/2.bitcaskdata",
    // ]
    let file_names: Vec<String> = std::fs::read_dir(&directory)
      .unwrap()
      .map(|entry| entry.unwrap().path().display().to_string())
      .collect();

    assert_eq!(3, file_names.len());

    for i in 0..3 {
      assert!(file_names
        .iter()
        .any(|file_name| file_name.contains(&format!("{}.bitcaskdata", i))));
    }

    Ok(())
  }

  #[quickcheck_macros::quickcheck]
  fn get_returns_none_if_key_does_not_exist_empty_bitcask(
    key: Vec<u8>,
  ) -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let directory = path(&temp_dir);

    let bitcask = default_bitcask(directory.clone())?;

    assert_eq!(None, bitcask.get(&key).unwrap());

    Ok(())
  }

  #[quickcheck_macros::quickcheck]
  fn get_returns_none_if_key_does_not_exist(
    keys: HashSet<Vec<u8>>,
    key: Vec<u8>,
  ) -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let directory = path(&temp_dir);

    let bitcask = default_bitcask(directory.clone())?;

    // Ignore this batch if key is in `keys`.
    if keys.contains(&key) {
      return Ok(());
    }

    for other_key in keys {
      assert_eq!(None, bitcask.get(&key).unwrap());
      bitcask.put(other_key.clone(), other_key)?;
      assert_eq!(None, bitcask.get(&key).unwrap());
    }

    Ok(())
  }

  #[quickcheck_macros::quickcheck]
  fn get_returns_value_associated_with_key(
    entries: Vec<(String, String)>,
  ) -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let directory = path(&temp_dir);

    let bitcask = default_bitcask(directory.clone())?;

    for (key, value) in entries {
      bitcask.put(bytes(&key), bytes(&value))?;

      let actual = String::from_utf8_lossy(&bitcask.get(&bytes(&key))?.unwrap()).to_string();

      assert_eq!(value, actual);
    }

    Ok(())
  }

  #[quickcheck_macros::quickcheck]
  fn delete_keys(key: Vec<u8>, value: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let directory = path(&temp_dir);

    let bitcask = default_bitcask(directory.clone())?;

    bitcask.put(key.clone(), value.clone())?;
    assert_eq!(Some(value), bitcask.get(&key)?);

    bitcask.delete(&key)?;
    assert_eq!(None, bitcask.get(&key)?);

    Ok(())
  }

  #[quickcheck_macros::quickcheck]
  fn list_keys(entries: HashSet<(Vec<u8>, Vec<u8>)>) -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let directory = path(&temp_dir);

    let bitcask = default_bitcask(directory.clone())?;

    let expected: HashSet<Vec<u8>> = entries.iter().map(|(key, _value)| key.clone()).collect();

    for (key, value) in entries {
      bitcask.put(key, value)?;
    }

    let actual: HashSet<Vec<u8>> = bitcask.list_keys().into_iter().collect();

    assert_eq!(expected, actual);

    Ok(())
  }

  #[test]
  fn bitcask_is_send_and_sync() {
    fn is_send_and_sync<T: Send + Sync>() {}
    is_send_and_sync::<Bitcask>();
  }
}
