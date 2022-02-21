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
  record_starts_at_position: u32,
  /// Timestamp of when the value was appended.
  timestamp: u32,
}

#[derive(Debug, PartialEq)]
enum Operation {
  Put,
  Delete,
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
  timestamp: u32,
  key: Vec<u8>,
  value: Vec<u8>,
}

/// Represents a Bitcask hint record. A hint file is created when a merge happens
/// and it contains a list of hint records.
/// Hint files are used to rebuild the index when a Bitcask instance is created.
///
/// A hint record looks like this on disk:
///                  ┌──────────────size of───────────────┐
///                  │                                    │
///                  │                                    │
/// ┌───────────┬────┴────┬───────────┬───────────┬───────▼──────┐
/// │ timestamp │ key_len │ value_len │ value_pos │      key     │
/// └───────────┴─────────┴───────────┴───────────┴──────────────┘
///      u32        u32       u32          u32      [u8; key_len]
#[derive(Debug)]
struct Hint {
  timestamp: u32,
  key: Vec<u8>,
  value_len: u32,
  record_starts_at_position: u32,
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
  /// Returns the file id that's in the file name.
  ///
  /// 0 would be returned for a file name like dir/bitcask.data.0, for example.
  fn file_id_from_file_path(path: &str) -> usize {
    path.split('.').last().unwrap().parse::<usize>().unwrap()
  }

  /// Returns the list of files created by Bitcask in `directory`.
  ///
  /// The list will usually look like this:
  ///
  /// dir/bitcask.data.1
  /// dir/bitcask.data.2
  /// dir/bitcask.data.3
  fn get_file_paths(directory: &str) -> std::io::Result<Vec<String>> {
    // Ensure `directory` exists.
    std::fs::create_dir_all(directory)?;

    Ok(
      std::fs::read_dir(directory)
        .unwrap()
        .filter(|entry| entry.is_ok())
        .map(|entry| entry.unwrap().path().display().to_string())
        .filter(|file_name| file_name.contains("bitcask"))
        .collect(),
    )
  }

  /// Returns a monotonically increasing unsigned id.
  ///
  /// The id is always the latest id increased by 1.
  fn get_file_id_for_new_active_file(files: &[String]) -> usize {
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
    files
      .iter()
      // Ensure we only take data files.
      .filter(|file_path| file_path.contains("bitcask.data"))
      .map(|file_path| Bitcask::file_id_from_file_path(file_path))
      .max()
      .map(|latest_id| latest_id + 1)
      .unwrap_or(0)
  }

  #[instrument(name = "Processing a hint record", skip_all)]
  fn process_hint_file<R: Read + Seek>(reader: &mut R) -> std::io::Result<Hint> {
    let timestamp = reader.read_u32::<LittleEndian>()?;
    info!(timestamp, "read timestamp from file");

    let key_len = reader.read_u32::<LittleEndian>()?;
    info!(key_len, "read key length from file");

    let value_len = reader.read_u32::<LittleEndian>()?;
    info!(value_len, "read value length from file");

    let record_starts_at_position = reader.read_u32::<LittleEndian>()?;
    info!(record_starts_at_position, "read record position from file");

    let mut buffer = Vec::with_capacity(key_len as usize);

    reader.take(key_len as u64).read_to_end(&mut buffer)?;

    let key = buffer;

    Ok(Hint {
      key,
      record_starts_at_position,
      timestamp,
      value_len,
    })
  }

  #[instrument(name = "Rebuilding index from files in the Bitcask directory", skip_all, fields(files = ?files))]
  fn build_index_from_files(files: &[String]) -> std::io::Result<HashMap<Vec<u8>, IndexEntry>> {
    let mut data_file_to_hint_file = HashMap::new();

    // Group data files to their hint file file, if there's one.
    // The map will look like this:
    // {
    //  "{{dir}}/bitcask.data.1": Some("{{dir}}/bitcask.hint.1"),
    //  "{{dir}}/bitcask.data.2":None,
    // }
    for file in files {
      if file.contains(".hint") {
        let key = file.replace(".hint", ".data");
        data_file_to_hint_file.insert(key, Some(file));
      } else if !data_file_to_hint_file.contains_key(file) {
        data_file_to_hint_file.insert(file.clone(), None);
      }
    }

    let mut index = HashMap::new();

    for (data_file_path, hint_file_path) in data_file_to_hint_file {
      info!(
        ?data_file_path,
        ?hint_file_path,
        "processing file and then merging it"
      );

      match hint_file_path {
        None => {
          info!(?hint_file_path, "processing data file");

          let file_id = Bitcask::file_id_from_file_path(&data_file_path);

          let mut reader = BufReader::new(File::open(&data_file_path)?);

          // Process every record in the file.
          loop {
            let record_starts_at_position = reader.seek(SeekFrom::Current(0))?;

            match Bitcask::process_record(&mut reader) {
              Err(err) => match err.kind() {
                std::io::ErrorKind::UnexpectedEof => {
                  break;
                }
                _ => return Err(err),
              },
              Ok(record) => {
                index.insert(
                  record.key,
                  IndexEntry {
                    file_id,
                    record_starts_at_position: record_starts_at_position as u32,
                    value_len: record.value.len() as u32,
                    timestamp: record.timestamp,
                  },
                );
              }
            }
          }
        }
        Some(path) => {
          info!(?hint_file_path, "processing hint file");

          let file_id = Bitcask::file_id_from_file_path(path);

          let mut reader = BufReader::new(File::open(path)?);

          // Process every record in the file.
          loop {
            let _ = reader.seek(SeekFrom::Current(0))?;

            match Bitcask::process_hint_file(&mut reader) {
              Err(err) => match err.kind() {
                std::io::ErrorKind::UnexpectedEof => {
                  break;
                }
                _ => return Err(err),
              },
              Ok(hint) => {
                let entry = IndexEntry {
                  file_id,
                  record_starts_at_position: hint.record_starts_at_position,
                  timestamp: hint.timestamp,
                  value_len: hint.value_len,
                };
                index.insert(hint.key, entry);
              }
            }
          }
        }
      }
    }

    Ok(index)
  }

  #[instrument(name = "Creating Bitcask instance", skip_all, fields(config = ?config))]
  pub fn new(config: Config) -> std::io::Result<Self> {
    let files_in_the_directory = Bitcask::get_file_paths(&config.directory)?;

    let file_id = Bitcask::get_file_id_for_new_active_file(&files_in_the_directory);

    let data_file_path = Path::new(&config.directory).join(format!("bitcask.data.{}", file_id));

    let active_file = OpenOptions::new()
      .read(true)
      .create(true)
      .append(config.writer)
      .open(data_file_path)?;

    let state = MutableState {
      active_file_id: file_id,
      index: Bitcask::build_index_from_files(&files_in_the_directory)?,
      active_file_size_in_bytes: 0,
      active_file,
      // Add 1 to take the file we just created into account.
      num_files_in_directory: files_in_the_directory.len() + 1,
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
    )
  )]
  pub fn put<V: AsRef<[u8]>>(&self, key: Vec<u8>, value: V) -> std::io::Result<()> {
    self.append(&key, value.as_ref(), Operation::Put)
  }

  /// Merges immutable data files into a new file that contains
  /// only the newest values for each key.
  #[instrument(name = "merge", skip_all)]
  fn merge(&self, state: &mut MutableState) -> std::io::Result<()> {
    let mut file_paths: Vec<(usize, String)> = Bitcask::get_file_paths(&self.config.directory)?
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

    info!(num_files = file_paths.len(), ?file_paths, "merging files");

    let mut mapping = HashMap::new();

    // Build a map from key to value starting from the oldest file
    // so we end up with the latest value for each key.
    for (_, file_path) in file_paths.iter() {
      info!(?file_path, "processing file and then merging it");

      let mut reader = BufReader::new(File::open(file_path)?);

      // Process every record in the file.
      loop {
        let _ = reader.seek(SeekFrom::Current(0))?;

        match Bitcask::process_record(&mut reader) {
          Err(err) => match err.kind() {
            std::io::ErrorKind::UnexpectedEof => {
              break;
            }
            _ => return Err(err),
          },
          Ok(record) => {
            // If we find a tombstone, it means the key was deleted at some point,
            // so we delete it from the map.
            if record.value == TOMBSTONE_VALUE {
              mapping.remove(&record.key);
            } else {
              mapping.insert(record.key, record.value);
            }
          }
        }
      }
    }

    // Create a temporary file.
    let merged_file_id = state.active_file_id - 1;
    let merged_file_path =
      Path::new(&self.config.directory).join(format!("bitcask.data.temp.{}", merged_file_id));

    let mut merged_file_writer = BufWriter::new(
      OpenOptions::new()
        .read(true)
        .create(true)
        .append(self.config.writer)
        .open(&merged_file_path)?,
    );

    let hint_file_path =
      Path::new(&self.config.directory).join(format!("bitcask.hint.temp.{}", merged_file_id));
    let mut hint_file_writer = BufWriter::new(
      OpenOptions::new()
        .read(true)
        .create(true)
        .append(self.config.writer)
        .open(&hint_file_path)?,
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

      let timestamp = chrono::Utc::now().timestamp() as u32;

      let record_starts_at_position;

      record_starts_at_position = merged_file_writer.seek(SeekFrom::End(0))? as u32;

      merged_file_writer.write_u32::<LittleEndian>(checksum)?;
      merged_file_writer.write_u32::<LittleEndian>(timestamp as u32)?;
      merged_file_writer.write_u32::<LittleEndian>(key_len as u32)?;
      merged_file_writer.write_u32::<LittleEndian>(value_len as u32)?;
      merged_file_writer.write_all(&key)?;
      merged_file_writer.write_all(&value)?;

      hint_file_writer.write_u32::<LittleEndian>(timestamp)?;
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

    // Take the new merged file into account.
    state.num_files_in_directory += 1;

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
          // Try to remove the hint file but ignore the error because it may not exist.
          // let _ = std::fs::remove_file(file_path.replace(".data", ".hint"));

          state.num_files_in_directory -= 1;
        }
      }
    }

    // Since everything worked out, promote the temporary file to a data file.
    self.promote_merged_file_to_data_file(&merged_file_path, &hint_file_path)?;

    Ok(())
  }

  #[instrument(
    name = "Promoting merged files",
    skip_all,
    fields(merged_file_path = ?merged_file_path, hint_file_path = ?hint_file_path)
  )]
  fn promote_merged_file_to_data_file(
    &self,
    merged_file_path: &Path,
    hint_file_path: &Path,
  ) -> std::io::Result<()> {
    std::fs::rename(
      merged_file_path,
      merged_file_path
        .to_str()
        .unwrap()
        .replace("bitcask.data.temp", "bitcask.data"),
    )?;

    std::fs::rename(
      hint_file_path,
      hint_file_path
        .to_str()
        .unwrap()
        .replace("bitcask.hint.temp", "bitcask.hint"),
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
    reader.seek(SeekFrom::Start(entry.record_starts_at_position as u64))?;

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
      timestamp,
      key,
      value,
    })
  }

  /// Appends a new entry to the active file.
  #[instrument(
      name = "Appending new entry to active file",
      skip_all,
      fields(
        key_utf8 = ?String::from_utf8_lossy(key), key = ?key,
        operation = ?operation
      )
    )]
  fn append(&self, key: &[u8], value: &[u8], operation: Operation) -> std::io::Result<()> {
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

    content_for_checksum.extend_from_slice(key);
    content_for_checksum.extend_from_slice(value);

    let checksum = crc32::checksum_ieee(&content_for_checksum);

    let timestamp = chrono::Utc::now().timestamp() as u32;

    let mut state = self.state.write().unwrap();

    let record_starts_at_position;

    {
      let mut writer = BufWriter::new(&mut state.active_file);

      record_starts_at_position = writer.seek(SeekFrom::End(0))? as u32;

      writer.write_u32::<LittleEndian>(checksum)?;
      writer.write_u32::<LittleEndian>(timestamp)?;
      writer.write_u32::<LittleEndian>(key_len as u32)?;
      writer.write_u32::<LittleEndian>(value_len as u32)?;
      writer.write_all(key)?;
      writer.write_all(value)?;

      writer.flush()?;
    }

    match operation {
      Operation::Delete => {
        state.index.remove(key);
      }
      Operation::Put => {
        let entry = IndexEntry {
          file_id: state.active_file_id,
          value_len: value_len as u32,
          timestamp,
          record_starts_at_position,
        };

        info!(?entry, "creating index entry");

        let _ = state.index.insert(key.to_vec(), entry);
      }
    }

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

  /// Forgets about the key.
  #[instrument(
    name = "delete",
    skip_all,
    fields(key_utf8 = ?String::from_utf8_lossy(key), key = ?key)
  )]
  pub fn delete(&self, key: &[u8]) -> std::io::Result<()> {
    self.append(key, TOMBSTONE_VALUE, Operation::Delete)
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
  use std::{collections::HashSet, sync::Arc, thread::JoinHandle, vec};

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
      bitcask.put(other_key.clone(), &other_key)?;
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
      bitcask.put(bytes(&key), &bytes(&value))?;

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

    bitcask.put(key.clone(), &value.clone())?;
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
      max_active_file_size_in_bytes: 32,
      merge_files_after_n_files_created: 3,
    })?;

    let key1 = bytes("key1");
    let value1 = bytes("value1");

    let key2 = bytes("key2");
    let value2 = bytes("value2");

    bitcask.put(key1.clone(), value1.clone())?;
    bitcask.put(key2.clone(), value2.clone())?;
    bitcask.delete(&key2)?;

    let file_paths: HashSet<String> = Bitcask::get_file_paths(&directory)?.into_iter().collect();

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

    // The key has been deleted before the merge, so it should be ignored when merging files.
    assert_eq!(None, bitcask.get(&key2)?);

    Ok(())
  }

  #[test_log::test]
  fn uses_files_in_the_directory_to_build_the_index_on_startup(
  ) -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let directory = path(&temp_dir);

    let bitcask = Bitcask::new(Config {
      directory: directory.clone(),
      writer: true,
      max_active_file_size_in_bytes: 32,
      merge_files_after_n_files_created: 3,
    })?;

    let key1 = bytes("key1");
    let value1 = bytes("value1");

    let key2 = bytes("key2");
    let value2 = bytes("value2");

    let key3 = bytes("key3");
    let value3 = bytes("value3");

    bitcask.put(key1.clone(), value1.clone())?;
    bitcask.put(key2.clone(), value2.clone())?;
    bitcask.put(key3.clone(), value3.clone())?;
    bitcask.delete(&key3)?;

    // Pretend the bitcask instance was recreated.
    let bitcask = Bitcask::new(Config {
      directory: directory.clone(),
      writer: true,
      max_active_file_size_in_bytes: 32,
      merge_files_after_n_files_created: 3,
    })?;

    // Ensure we can still access the keys we added before the restart.
    assert_eq!(Some(value1), bitcask.get(&key1)?);
    assert_eq!(Some(value2), bitcask.get(&key2)?);
    assert_eq!(None, bitcask.get(&key3)?);

    Ok(())
  }

  #[quickcheck_macros::quickcheck]
  fn a_few_threads(entries: Vec<(Vec<u8>, Vec<u8>)>) -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let directory = path(&dir);

    let bitcask = Arc::new(Bitcask::new(Config {
      directory,
      writer: true,
      max_active_file_size_in_bytes: 5,
      merge_files_after_n_files_created: 3,
    })?);

    let handles: Vec<JoinHandle<()>> = entries
      .iter()
      .cloned()
      .map(|(key, value)| {
        let bitcask = Arc::clone(&bitcask);

        std::thread::spawn(move || {
          bitcask.put(key.clone(), value).expect("error putting key");
          bitcask.delete(&key).expect("error deleting key");
        })
      })
      .collect();

    for handle in handles {
      handle.join().expect("error joining thread");
    }

    for (key, _) in entries {
      assert_eq!(None, bitcask.get(&key)?);
    }

    Ok(())
  }

  #[test]
  fn bitcask_is_send_and_sync() {
    fn is_send_and_sync<T: Send + Sync>() {}
    is_send_and_sync::<Bitcask>();
  }
}
