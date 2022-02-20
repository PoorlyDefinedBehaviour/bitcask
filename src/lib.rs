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
  /// The number of files in the Bitcask directory.
  /// Used to know when we need to perform merging.
  num_files_in_directory: usize,
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
  /// We will perform merging after the Bitcask directory has a number of files
  /// thats >= `merge_files_after_n_files_created`.
  pub merge_files_after_n_files_created: usize,
}

impl Bitcask {
  fn file_id_from_file_path(path: &str) -> usize {
    path.split('.').last().unwrap().parse::<usize>().unwrap()
  }

  fn get_file_paths(directory: &str) -> Vec<String> {
    std::fs::read_dir(directory)
      .unwrap()
      .filter(|entry| entry.is_ok())
      .map(|entry| entry.unwrap().path().display().to_string())
      .filter(|file_name| file_name.contains("bitcask"))
      .collect()
  }
  fn get_file_id(directory: &str) -> std::io::Result<(usize, usize)> {
    // Ensure `directory` exists.
    std::fs::create_dir_all(directory)?;

    let file_names: Vec<String> = Bitcask::get_file_paths(directory)
      .into_iter()
      // Ensure we only take data files.
      .filter(|file_name| file_name.contains("bitcask.data"))
      .collect();

    // Given a directory that has a bunch of Bitcask files, we want to find out which file is the latest
    // and we can do that by using the index that is added to each file.
    //
    // If we have the following files:
    //
    // dir/bitcask.data.0
    // dir/bitcask.data.1
    // dir/bitcask.data.2
    //
    // we know that the latest file is file number 2.
    let latest_id = file_names
      .iter()
      .map(|file_path| file_path.split('.').last().unwrap().to_string())
      .map(|offset| offset.parse::<usize>().unwrap())
      .max();

    let new_id = match latest_id {
      None => 0,
      Some(id) => id + 1,
    };

    Ok((new_id, file_names.len()))
  }

  #[instrument(name = "Creating Bitcask instance", skip_all, fields(config = ?config))]
  pub fn new(config: Config) -> std::io::Result<Self> {
    let (file_id, num_files_in_directory) = Bitcask::get_file_id(&config.directory)?;

    let data_file_path = Path::new(&config.directory).join(format!("bitcask.data.{}", file_id));

    let active_file = OpenOptions::new()
      .read(true)
      .create(true)
      .append(config.writer)
      .open(data_file_path)?;

    let state = MutableState {
      active_file_id: file_id,
      index: HashMap::new(),
      active_file_size_in_bytes: 0,
      active_file,
      // Add 1 to take the file we just created into account.
      num_files_in_directory: num_files_in_directory + 1,
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

    if state.active_file_size_in_bytes >= self.config.max_active_file_size_in_bytes {
      info!(
        active_file_id = state.active_file_id,
        active_file_size_in_bytes = state.active_file_size_in_bytes,
        max_active_file_size_in_bytes = self.config.max_active_file_size_in_bytes,
        "active file has reached its maximum size. creating a new active file"
      );

      state.active_file_id += 1;

      let data_file_path =
        Path::new(&self.config.directory).join(format!("bitcask.data.{}", state.active_file_id));

      state.active_file = OpenOptions::new()
        .read(true)
        .create(true)
        .append(self.config.writer)
        .open(data_file_path)?;

      state.active_file_size_in_bytes = 0;
      state.num_files_in_directory += 1;

      info!(
        active_file_id = state.active_file_id,
        "new active file created"
      );
    }

    if state.num_files_in_directory >= self.config.merge_files_after_n_files_created {
      info!(
        num_files_in_directory = state.num_files_in_directory,
        merge_files_after_n_files_created = self.config.merge_files_after_n_files_created,
        "directory reached the maximum number of files"
      );

      self.merge(&mut state)?;
    }

    Ok(())
  }

  /// Merges immutable data files into a new file that contains
  /// only the newest values for each key.
  #[instrument(name = "merge", skip_all)]
  fn merge(&self, state: &mut MutableState) -> std::io::Result<()> {
    // dir/bitcask.data.0
    // dir/bitcask.data.2
    // dir/bitcask.data.1
    let mut file_paths: Vec<(usize, String)> = Bitcask::get_file_paths(&self.config.directory)
      .into_iter()
      // Ensure we only take data files that were created by Bitcask.
      .filter(|file_path| file_path.contains("bitcask.data"))
      .map(|file_path| (Bitcask::file_id_from_file_path(&file_path), file_path))
      .filter(|(file_id, _)| *file_id != state.active_file_id)
      .collect();

    // Sort file paths by the file id:
    // dir/bitcask.data.0
    // dir/bitcask.data.1
    // dir/bitcask.data.2
    file_paths.sort_unstable_by_key(|(file_id, _file_path)| *file_id);

    info!(num_files = file_paths.len(), "merging files");

    let mut mapping = HashMap::new();

    // Build a map from key to value starting from the oldest file
    // so we end up with the latest value for each key.
    for (_, file_path) in file_paths.iter() {
      info!(?file_path, "processing file and then merging it");

      let mut reader = BufReader::new(File::open(file_path)?);
      match Bitcask::process_record(&mut reader) {
        Err(err) => match err.kind() {
          std::io::ErrorKind::UnexpectedEof => {
            break;
          }
          _ => return Err(err),
        },
        Ok(record) => {
          mapping.insert(record.key, record.value);
        }
      }
    }

    // Create a temporary file.
    let merged_file_id = state.active_file_id - 1;
    let merged_file_path =
      Path::new(&self.config.directory).join(format!("bitcask.data.merged.{}", merged_file_id));

    let mut merged_file_writer = BufWriter::new(
      OpenOptions::new()
        .read(true)
        .create(true)
        .append(self.config.writer)
        .open(&merged_file_path)?,
    );

    let mut hint_file_writer = BufWriter::new(
      OpenOptions::new()
        .read(true)
        .create(true)
        .append(self.config.writer)
        .open(Path::new(&self.config.directory).join(format!("bitcask.hint.{}", merged_file_id)))?,
    );

    let mut index = HashMap::new();

    // Create a file that contains only the latest value for each key.
    for (key, value) in mapping {
      let key_len = key.len();
      let value_len = value.len();

      let mut content_for_checksum = Vec::with_capacity(key_len + value_len);

      content_for_checksum.extend_from_slice(&key);
      content_for_checksum.extend_from_slice(&value);

      let checksum = crc32::checksum_ieee(&content_for_checksum);

      let timestamp = chrono::Utc::now().timestamp();

      let record_starts_at_position;

      record_starts_at_position = merged_file_writer.seek(SeekFrom::End(0))?;

      merged_file_writer.write_u32::<LittleEndian>(checksum)?;
      merged_file_writer.write_u32::<LittleEndian>(timestamp as u32)?;
      merged_file_writer.write_u32::<LittleEndian>(key_len as u32)?;
      merged_file_writer.write_u32::<LittleEndian>(value_len as u32)?;
      merged_file_writer.write_all(&key)?;
      merged_file_writer.write_all(&value)?;

      hint_file_writer.write_u32::<LittleEndian>(timestamp as u32)?;
      hint_file_writer.write_u32::<LittleEndian>(key_len as u32)?;
      hint_file_writer.write_u32::<LittleEndian>(value_len as u32)?;
      hint_file_writer.write_u32::<LittleEndian>(record_starts_at_position as u32)?;
      hint_file_writer.write_all(&key)?;

      let entry = IndexEntry {
        file_id: merged_file_id,
        value_len: value_len as u32,
        timestamp,
        record_starts_at_position,
      };

      index.insert(key, entry);
    }

    merged_file_writer.flush()?;
    hint_file_writer.flush()?;

    state.index = index;

    for (_file_id, file_path) in file_paths.iter() {
      info!(?file_path, "removing old data file that has been merged");
      // This error is not a show stopper since we can just reprocess the file after in the next merge.
      match std::fs::remove_file(file_path) {
        Err(err) => {
          error!(error = ?err, ?file_path, "error removing old data file that has been merged");
          // Do not delete newer files to avoid ending up in a invalid state when we try to merge again.
          break;
        }
        Ok(_) => {
          state.num_files_in_directory -= 1;
        }
      }
    }

    // Since everything worked out, promote the temporary file to a data file.
    self.promote_merged_file_to_data_file(&merged_file_path, merged_file_id)?;

    Ok(())
  }

  #[instrument(
    name = "Promoting merged file to data file",
    skip_all,
    fields(merged_file_path = ?merged_file_path, merged_file_id = merged_file_id)
  )]
  fn promote_merged_file_to_data_file(
    &self,
    merged_file_path: &Path,
    merged_file_id: usize,
  ) -> std::io::Result<()> {
    std::fs::rename(
      merged_file_path,
      Path::new(&self.config.directory).join(format!("bitcask.data.{}", merged_file_id)),
    )
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

    info!(?entry, "entry found");

    let file_path =
      Path::new(&self.config.directory).join(format!("bitcask.data.{}", entry.file_id));

    info!(?file_path, "opening data file");

    let mut reader = match File::open(&file_path) {
      Err(e) => {
        error!(error = ?e, "data file not found");
        return Err(e);
      }
      Ok(file) => BufReader::new(file),
    };

    // Move the reader to where the record starts.
    reader.seek(SeekFrom::Start(entry.record_starts_at_position))?;

    let record = Bitcask::process_record(&mut reader)?;

    info!(?record, "record found");

    Ok(Some(record.value))
  }

  // Reads a record into memory.
  #[instrument(name = "Processing record", skip_all)]
  fn process_record<R: Read + Seek>(reader: &mut R) -> std::io::Result<Record> {
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
      merge_files_after_n_files_created: 32,
    })
  }

  /// Returns a list of file names like:
  ///
  /// [
  ///  "bitcask.data.0",
  ///  "bitcask.data.1",
  ///  "bitcask.data.1",
  /// ]
  fn get_file_names(dir: &str) -> Vec<String> {
    std::fs::read_dir(dir)
      .unwrap()
      .map(|entry| entry.unwrap().path().display().to_string())
      .map(|s| s.split("/").last().unwrap().to_string())
      .collect()
  }

  #[test_log::test]
  fn each_bitcask_instance_creates_a_new_data_file() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let directory = path(&temp_dir);

    for i in 0..3 {
      let bitcask = default_bitcask(directory.clone())?;
      assert_eq!(i, bitcask.state.read().unwrap().active_file_id);
    }

    let file_names = get_file_names(&directory);

    assert_eq!(3, file_names.len());

    for i in 0..3 {
      assert!(file_names.contains(&format!("bitcask.data.{}", i)));
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

  #[test_log::test]
  fn closes_active_file_that_becomes_too_large() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let directory = path(&temp_dir);

    let bitcask = Bitcask::new(Config {
      directory: directory.clone(),
      writer: true,
      max_active_file_size_in_bytes: 5,
      merge_files_after_n_files_created: 32,
    })?;

    assert_eq!(0, bitcask.state.read().unwrap().active_file_id);
    assert_eq!(1, bitcask.state.read().unwrap().num_files_in_directory);

    bitcask.put(bytes("key"), bytes("value_with_more_than_5_bytes"))?;

    assert_eq!(1, bitcask.state.read().unwrap().active_file_id);
    assert_eq!(0, bitcask.state.read().unwrap().active_file_size_in_bytes);
    assert_eq!(2, bitcask.state.read().unwrap().num_files_in_directory);

    let file_names = HashSet::from_iter(get_file_names(&directory));

    let expected: HashSet<String> = HashSet::from_iter(vec![
      String::from("bitcask.data.0"),
      String::from("bitcask.data.1"),
    ]);

    assert_eq!(expected, file_names);

    Ok(())
  }

  #[test_log::test]
  fn merges_immutable_files_when_threshold_is_surpassed() -> Result<(), Box<dyn std::error::Error>>
  {
    let temp_dir = tempfile::tempdir()?;
    let directory = path(&temp_dir);

    let bitcask = Bitcask::new(Config {
      directory: directory.clone(),
      writer: true,
      max_active_file_size_in_bytes: 1,
      merge_files_after_n_files_created: 3,
    })?;

    let key1 = bytes("key1");
    let value1 = bytes("value1");

    let key2 = bytes("key2");
    let value2 = bytes("value2");

    bitcask.put(key1.clone(), value1.clone())?;
    bitcask.put(key2.clone(), value2.clone())?;

    let file_paths: HashSet<String> = Bitcask::get_file_paths(&directory).into_iter().collect();

    let expected: HashSet<String> = vec![
      format!("{}/bitcask.data.1", &directory),
      format!("{}/bitcask.hint.1", &directory),
      format!("{}/bitcask.data.2", &directory),
    ]
    .into_iter()
    .collect();

    assert_eq!(expected, file_paths);

    // Ensure we can still accesses the keys we added before the merge.
    assert_eq!(Some(value1), bitcask.get(&key1)?);
    assert_eq!(Some(value2), bitcask.get(&key2)?);

    Ok(())
  }

  #[test]
  fn bitcask_is_send_and_sync() {
    fn is_send_and_sync<T: Send + Sync>() {}
    is_send_and_sync::<Bitcask>();
  }
}
