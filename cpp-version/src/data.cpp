#include "data.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cstdio>

void download_with_requests(const std::string &url, const std::string &out_path)
{
    // Use system command for download (cross-platform fallback)
    // On Windows, use PowerShell; on Unix, use curl/wget
#ifdef _WIN32
    std::string cmd = "powershell -Command \"Invoke-WebRequest -Uri '" + url +
                      "' -OutFile '" + out_path +
                      "' -UserAgent 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'\"";
#else
    std::string cmd = "curl -L -o '" + out_path + "' -A 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' '" + url + "'";
#endif
    std::system(cmd.c_str());
}

std::vector<std::pair<std::string, std::string>> get_raw_data(int limit, int max_len)
{
    // Download dataset if not present
    std::ifstream check("ben.txt");
    if (!check.is_open())
    {
        std::cout << "Downloading dataset..." << std::endl;
        download_with_requests("https://www.manythings.org/anki/ben-eng.zip", "b.zip");

        // Extract zip
#ifdef _WIN32
        std::system("powershell -Command \"Expand-Archive -Path 'b.zip' -DestinationPath '.' -Force\"");
#else
        std::system("unzip -o b.zip");
#endif
    }
    else
    {
        check.close();
    }

    // Parse dataset
    std::vector<std::pair<std::string, std::string>> pairs;
    std::ifstream file("ben.txt");

    if (!file.is_open())
    {
        std::cout << "Error: ben.txt not found." << std::endl;
        return pairs;
    }

    std::string line;
    int i = 0;

    while (std::getline(file, line))
    {
        // Tab-separated format
        auto pos = line.find('\t');
        if (pos != std::string::npos)
        {
            std::string en = line.substr(0, pos);
            std::string bn = line.substr(pos + 1);

            // Handle second tab (attribution field)
            auto pos2 = bn.find('\t');
            if (pos2 != std::string::npos)
            {
                bn = bn.substr(0, pos2);
            }

            // Trim whitespace
            auto trim = [](std::string &s)
            {
                s.erase(0, s.find_first_not_of(" \t\r\n"));
                s.erase(s.find_last_not_of(" \t\r\n") + 1);
            };
            trim(en);
            trim(bn);

            // Filter by sentence length (word count).
            // Matches Python str.split(): any whitespace run separates words,
            // leading/trailing whitespace ignored, empty string -> 0 words.
            auto word_count = [](const std::string &s)
            {
                int count = 0;
                bool in_word = false;
                for (unsigned char c : s)
                {
                    if (std::isspace(c))
                    {
                        in_word = false;
                    }
                    else if (!in_word)
                    {
                        in_word = true;
                        ++count;
                    }
                }
                return count;
            };

            if (word_count(en) < max_len && word_count(bn) < max_len)
            {
                pairs.push_back({en, bn});
            }
        }
        if (i >= limit)
            break;
        i++;
    }

    return pairs;
}
