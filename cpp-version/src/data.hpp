#pragma once
#include <string>
#include <vector>
#include <utility>

/**
 * @brief Data loading utilities for the ManyThings Ben-Eng dataset.
 */

/**
 * @brief Download file with proper headers to avoid 403 errors.
 * @param url URL to download from
 * @param out_path Local path to save file
 */
void download_with_requests(const std::string &url, const std::string &out_path);

/**
 * @brief Download and parse the English-Bengali dataset.
 * @param limit Maximum number of sentence pairs to load (default 50000)
 * @param max_len Maximum sentence length in words to include (default 64)
 * @return Vector of (english, bengali) sentence pairs
 */
std::vector<std::pair<std::string, std::string>> get_raw_data(
    int limit = 50000, int max_len = 64);
