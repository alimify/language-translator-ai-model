//! Data loading utilities.
//!
//! Matches Python: data.py

use std::fs;
use std::io::{ BufRead, BufReader, Write };
use std::path::Path;

/// Download file with proper headers to avoid 403 errors.
/// Matches Python: def download_with_requests(url, out_path)
fn download_with_requests(url: &str, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let resp = ureq
        ::get(url)
        .set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        .set("Accept", "*/*")
        .call()?;

    let mut bytes = Vec::new();
    resp.into_reader().read_to_end(&mut bytes)?;

    let mut file = fs::File::create(out_path)?;
    file.write_all(&bytes)?;

    Ok(())
}

/// Download and parse the English-Bengali dataset.
///
/// Matches Python: def get_raw_data(limit=50000, max_len=64)
///
/// # Arguments
/// * `limit` - Maximum number of sentence pairs to load
/// * `max_len` - Maximum sentence length (in words)
///
/// # Returns
/// Vector of (english, bengali) pairs
pub fn get_raw_data(limit: usize, max_len: usize) -> Vec<(String, String)> {
    // Download dataset if not present
    if !Path::new("ben.txt").exists() {
        println!("Downloading dataset...");
        if
            let Err(e) = download_with_requests(
                "https://www.manythings.org/anki/ben-eng.zip",
                "b.zip"
            )
        {
            eprintln!("Download failed: {}", e);
            return Vec::new();
        }

        // Extract zip
        if let Ok(file) = fs::File::open("b.zip") {
            if let Ok(mut archive) = zip::ZipArchive::new(file) {
                for i in 0..archive.len() {
                    if let Ok(mut entry) = archive.by_index(i) {
                        let name = entry.name().to_string();
                        if let Ok(mut out) = fs::File::create(&name) {
                            std::io::copy(&mut entry, &mut out).ok();
                        }
                    }
                }
            }
        }
    }

    let mut pairs = Vec::new();

    let file = match fs::File::open("ben.txt") {
        Ok(f) => f,
        Err(_) => {
            eprintln!("Error: ben.txt not found.");
            return pairs;
        }
    };

    let reader = BufReader::new(file);

    for (i, line) in reader.lines().enumerate() {
        if let Ok(line) = line {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                let en = parts[0].trim().to_string();
                let bn = parts[1].trim().to_string();

                // Filter by word count
                if
                    en.split_whitespace().count() < max_len &&
                    bn.split_whitespace().count() < max_len
                {
                    pairs.push((en, bn));
                }
            }
        }
        if i >= limit {
            break;
        }
    }

    pairs
}
