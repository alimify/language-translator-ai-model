package translator;

import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.util.*;

/**
 * Data loading utilities.
 * Matches Python: data.py
 */
public class DataLoader {

    /**
     * Download file with User-Agent header.
     * Matches Python: def download_with_requests(url, out_path)
     */
    public static void download_with_requests(String url, String out_path) {
        try {
            // Use system command for download (cross-platform)
            String os = System.getProperty("os.name").toLowerCase();
            String cmd;
            if (os.contains("win")) {
                cmd = "powershell -Command \"Invoke-WebRequest -Uri '" + url +
                        "' -OutFile '" + out_path +
                        "' -UserAgent 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'\"";
            } else {
                cmd = "curl -L -o '" + out_path + "' -A 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' '" + url + "'";
            }
            Runtime.getRuntime().exec(new String[] { "bash", "-c", cmd }).waitFor();
        } catch (Exception e) {
            System.err.println("Download failed: " + e.getMessage());
        }
    }

    /**
     * Load English-Bengali sentence pairs.
     * Matches Python: def get_raw_data(limit=50000, max_len=64)
     */
    public static List<String[]> get_raw_data(int limit) {
        return get_raw_data(limit, 64);
    }

    public static List<String[]> get_raw_data(int limit, int max_len) {
        // Download dataset if not present
        if (!Files.exists(Paths.get("ben.txt"))) {
            System.out.println("Downloading dataset...");
            download_with_requests("https://www.manythings.org/anki/ben-eng.zip", "b.zip");

            // Extract zip
            try {
                String os = System.getProperty("os.name").toLowerCase();
                if (os.contains("win")) {
                    Runtime.getRuntime().exec(new String[] {
                            "powershell", "-Command",
                            "Expand-Archive -Path 'b.zip' -DestinationPath '.' -Force"
                    }).waitFor();
                } else {
                    Runtime.getRuntime().exec(new String[] { "unzip", "-o", "b.zip" }).waitFor();
                }
            } catch (Exception e) {
                System.err.println("Extraction failed: " + e.getMessage());
            }
        }

        // Parse dataset
        List<String[]> pairs = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(Paths.get("ben.txt"))) {
            String line;
            int i = 0;

            while ((line = reader.readLine()) != null) {
                String[] parts = line.split("\t");
                if (parts.length >= 2) {
                    String en = parts[0].trim();
                    String bn = parts[1].trim();

                    // Filter by word count. Matches Python str.split():
                    // empty string -> 0 words (Java "".split gives [""]).
                    int en_words = en.isEmpty() ? 0 : en.split("\\s+").length;
                    int bn_words = bn.isEmpty() ? 0 : bn.split("\\s+").length;
                    if (en_words < max_len && bn_words < max_len) {
                        pairs.add(new String[] { en, bn });
                    }
                }
                if (i >= limit)
                    break;
                i++;
            }
        } catch (IOException e) {
            System.out.println("Error: ben.txt not found.");
        }

        return pairs;
    }
}
