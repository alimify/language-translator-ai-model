/**Data loading utilities for the ManyThings Ben-Eng dataset.*/
import fs from "fs";
import axios from "axios";
import AdmZip from "adm-zip";

/**Download file with proper headers to avoid 403 errors.
 * 
 * Args:
 *     url: URL to download from
 *     out_path: Local path to save file
 */
async function download_with_requests(url, out_path) {
    const headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        Accept: "*/*",
    };
    const response = await axios({
        method: "get",
        url: url,
        headers: headers,
        responseType: "arraybuffer",
        timeout: 30000,
    });
    fs.writeFileSync(out_path, Buffer.from(response.data));
}

/**Download and parse the English-Bengali dataset.
 * 
 * Args:
 *     limit: Maximum number of sentence pairs to load
 *     max_len: Maximum sentence length (in words) to include
 *     
 * Returns:
 *     Array of [english, bengali] sentence pairs
 */
async function get_raw_data(limit = 50000, max_len = 64) {
    // Download dataset if not present
    if (!fs.existsSync("ben.txt")) {
        console.log("Downloading dataset...");
        await download_with_requests(
            "https://www.manythings.org/anki/ben-eng.zip",
            "b.zip"
        );
        const zip = new AdmZip("b.zip");
        zip.extractAllTo(".", true);
    }

    // Parse dataset
    const pairs = [];
    if (!fs.existsSync("ben.txt")) {
        console.log("Error: ben.txt not found.");
        return [];
    }

    const content = fs.readFileSync("ben.txt", "utf8");
    const lines = content.split("\n");

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const parts = line.split("\t");
        if (parts.length >= 2) {
            const en = parts[0].trim();
            const bn = parts[1].trim();
            // Filter by sentence length. Matches Python str.split(): any
            // whitespace run separates words; empty string -> 0 words.
            const word_count = (s) =>
                s.trim() === "" ? 0 : s.trim().split(/\s+/).length;
            if (word_count(en) < max_len && word_count(bn) < max_len) {
                pairs.push([en, bn]);
            }
        }
        if (i >= limit) {
            break;
        }
    }

    return pairs;
}

export { download_with_requests, get_raw_data };
