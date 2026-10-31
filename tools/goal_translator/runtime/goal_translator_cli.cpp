// goal_translator_cli.cpp - standalone driver for the NEON Goal Translator (neon_goal_translator.hpp).
//
// Build (from this repo's root, NEON = your NANITY-NEON checkout):
//   g++ -std=c++20 -O2 -pthread -mavx2 -mfma -I $NEON/runtime tools/goal_translator/runtime/goal_translator_cli.cpp -o goal_translator
//
//   goal_translator MODEL.gguf info
//   goal_translator MODEL.gguf translate "walk right then left"          -> one JSON line (the windows as [turn_bin, speed_bin] or null)
//   goal_translator MODEL.gguf logits 257,119,97,...                      -> next-token logits after that token list, one CSV row (tests/test_goal_translator_neon.py)
//   goal_translator MODEL.gguf --interactive                              JSON lines on stdin ({"op":"goal_translate","text":...}), one JSON line out per request
// Options: --threads N
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "neon_goal_translator.hpp"

int main(int argc, char** argv) try {
    if (argc < 3) { std::fprintf(stderr, "usage: %s MODEL.gguf (info|translate TEXT|logits IDS|--interactive) [--threads N]\n", argv[0]); return 2; }
    int threads = 1; std::vector<std::string> a;
    for (int i = 1; i < argc; ++i) { std::string s = argv[i]; if (s == "--threads" && i + 1 < argc) threads = std::max(1, std::atoi(argv[++i])); else a.push_back(s); }
    util::ThreadPool pool(threads);
    goaltr::GoalTranslator gt; gt.load(a[0], &pool);
    const std::string cmd = a.size() > 1 ? a[1] : "info";
    if (cmd == "info") { std::cout << gt.info() << "\n"; return 0; }
    if (cmd == "translate") {
        if (a.size() < 3) { std::fprintf(stderr, "translate needs a text\n"); return 2; }
        std::string text = a[2], esc; for (char c : text) { if (c == '"' || c == '\\') esc += '\\'; esc += c; }
        std::cout << gt.handle(json::parse("{\"op\":\"goal_translate\",\"text\":\"" + esc + "\"}")) << "\n"; return 0;
    }
    if (cmd == "logits") {
        if (a.size() < 3) { std::fprintf(stderr, "logits needs a comma separated token list\n"); return 2; }
        std::string ids = a[2]; for (char& c : ids) if (c == ',') c = ' ';
        std::istringstream ss(ids); long v; std::string arr = "[";
        while (ss >> v) { if (arr.size() > 1) arr += ","; arr += std::to_string(v); }
        const std::string r = gt.handle(json::parse("{\"op\":\"goal_logits\",\"tokens\":" + arr + "]}"));
        std::cout << r << "\n"; return r.find("\"ok\":true") != std::string::npos ? 0 : 1;
    }
    if (cmd == "--interactive") {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
            try { std::cout << gt.handle(json::parse(line)) << "\n" << std::flush; }
            catch (const std::exception& e) { std::cout << "{\"ok\":false,\"error\":\"bad request\"}\n" << std::flush; }
        }
        return 0;
    }
    std::fprintf(stderr, "unknown command '%s'\n", cmd.c_str()); return 2;
} catch (const std::exception& e) { std::fprintf(stderr, "[goal_translator] %s\n", e.what()); return 1; }
