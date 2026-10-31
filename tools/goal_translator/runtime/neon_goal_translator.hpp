// neon_goal_translator.hpp - the NANITY Goal Translator (spec NANITY_GOAL_TRANSLATOR_SPEC.md) in NEON: text bytes in, a goal plan out.
//
// The model is an ordinary NANITY v1 GGUF (general.architecture = "nanity"; training/goal_translator/export_goal_translator.py), run by the forward pass NEON already has
// (fwd::transformer_forward). What is new here is only what the spec puts around it:
//   * the input is raw bytes:  [BOS] b0 b1 ... [SEP]       (no text tokenizer; > max_input_bytes is an error, never a truncation)
//   * the output grammar:      WIN (TURN SPEED | UNSPEC) ... END      applied as a logit mask, greedy, so a malformed plan cannot be produced
//   * reaching max_windows without END is an error, not a truncation
// It emits (turn bin, speed bin) per window or null (UNSPEC). Turning the bins into the prior's four goal numbers (arc_goal, clamp to goal_stats) is the caller's job and is
// done with the numbers this class reports (goal_stats, window timing): see UI/goal_plan.py in NERVA, and info() below. Token layout and conventions: training/goal_translator/gt_conventions.py
// (checked at load against the file's own nanity.goal_translator.* keys, layout_version 1).
//
// JSON ops (handle()):   {"op":"info"}
//                        {"op":"goal_translate","text":"walk left then right"}  -> {"ok":true,"windows":[[turn_bin,speed_bin]|null,...],"count":N,"tokens":[...]}
//                        {"op":"goal_logits","tokens":[...]}                    -> {"ok":true,"logits":[V floats]}   (next-token logits after the full token list; for tests)
// Errors are {"ok":false,"error":"..."} (the caller shows them; there is no fallback in here).
//
// Header-only. Includes the same core headers NEON-3.cpp does: compile with -I <NANITY-NEON>/runtime.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/rawllm_common.hpp"
#include "core/rawllm_util.hpp"
#include "core/rawllm_json.hpp"
#include "core/rawllm_loader.hpp"
#include "core/rawllm_forward.hpp"

namespace goaltr {

class GoalTranslator {
public:
    // ---- the token layout of the file (read from nanity.goal_translator.*; must equal gt_conventions.py LAYOUT_VERSION 1) ----
    uint32_t layout_version = 0, n_turn_bins = 0, n_speed_bins = 0, max_windows = 0, max_input_bytes = 0, vocab = 0;
    int32_t PAD = 0, BOS = 0, SEP = 0, WIN = 0, UNSPEC = 0, END = 0, TURN_BASE = 0, SPEED_BASE = 0;
    float turn_max = 0, speed_cruise = 0, speed_max = 0, dt = 0;
    uint32_t window_W = 0, window_stride = 0;

    bool ready() const { return loaded_; }

    void load(const std::string& path, util::ThreadPool* pool) {
        pool_ = pool;
        file_ = std::make_unique<loader::GGUFLoader>();
        file_->open(path);
        if (file_->meta_string("general.architecture", "") != "nanity" || file_->meta_number("nanity.goal_translator.layout_version", -1) < 0)
            throw std::runtime_error("goal translator: " + path + " is not a Goal Translator GGUF (general.architecture nanity + nanity.goal_translator.* keys)");
        auto num = [&](const char* k) { return file_->meta_number(std::string("nanity.goal_translator.") + k, std::nan("")); };
        auto u = [&](const char* k) { double v = num(k); if (std::isnan(v)) throw std::runtime_error(std::string("goal translator: missing key nanity.goal_translator.") + k); return v; };
        layout_version = (uint32_t)u("layout_version");
        if (layout_version != 1) throw std::runtime_error("goal translator: token layout version " + std::to_string(layout_version) + " is not supported by this NEON (it knows 1)");
        n_turn_bins = (uint32_t)u("n_turn_bins"); n_speed_bins = (uint32_t)u("n_speed_bins"); max_windows = (uint32_t)u("max_windows");
        max_input_bytes = (uint32_t)u("max_input_bytes"); vocab = (uint32_t)u("vocab_size");
        PAD = (int32_t)u("pad"); BOS = (int32_t)u("bos"); SEP = (int32_t)u("sep"); WIN = (int32_t)u("win"); UNSPEC = (int32_t)u("unspec"); END = (int32_t)u("end");
        TURN_BASE = (int32_t)u("turn_base"); SPEED_BASE = (int32_t)u("speed_base");
        turn_max = (float)u("turn_max"); speed_cruise = (float)u("speed_cruise"); speed_max = (float)u("speed_max");
        window_W = (uint32_t)u("window_W"); window_stride = (uint32_t)u("window_stride"); dt = (float)u("window_dt");
        if (TURN_BASE + (int32_t)n_turn_bins != SPEED_BASE || SPEED_BASE + (int32_t)n_speed_bins > (int32_t)vocab || END >= TURN_BASE)
            throw std::runtime_error("goal translator: the token ids in the file are not a consistent layout");
        cfg_ = engine::Config();
        cfg_.n_threads = pool ? 0 : 1;
        file_->validate_config(cfg_);
        if (cfg_.n_vocab != vocab) throw std::runtime_error("goal translator: n_vocab " + std::to_string(cfg_.n_vocab) + " != vocab_size key " + std::to_string(vocab));
        if (cfg_.ctx_len < 1 + max_input_bytes + 1 + 3 * max_windows + 1)
            throw std::runtime_error("goal translator: the model's context_length is shorter than the longest input plus the longest plan");
        mw_ = std::make_unique<fwd::ModelWeights>(fwd::ModelWeights::build(*file_, cfg_));
        loaded_ = true;
    }

    std::string info() const {
        std::ostringstream o;
        o << "{\"ok\":true,\"goal_translator\":{\"layout_version\":" << layout_version << ",\"turn_bins\":" << n_turn_bins << ",\"speed_bins\":" << n_speed_bins
          << ",\"max_windows\":" << max_windows << ",\"max_input_bytes\":" << max_input_bytes << ",\"vocab\":" << vocab << ",\"n_embd\":" << cfg_.n_embd << ",\"n_layer\":" << cfg_.n_layer
          << ",\"goal_stats\":{\"turn_max\":" << turn_max << ",\"speed_cruise\":" << speed_cruise << ",\"speed_max\":" << speed_max << "},\"window\":{\"W\":" << window_W
          << ",\"stride\":" << window_stride << ",\"dt\":" << dt << "}}}";
        return o.str();
    }

    // Greedy, grammar-masked decoding. Returns the OUTPUT tokens (ending in END). Throws std::runtime_error on an over-long input, the window cap, or a cut-off output.
    std::vector<int32_t> decode(const std::string& text, std::vector<float>* last_logits = nullptr) const {
        need();
        if (text.size() > max_input_bytes)
            throw std::runtime_error("text is " + std::to_string(text.size()) + " bytes; the limit is " + std::to_string(max_input_bytes) + " (longer text is rejected, not truncated)");
        std::vector<int32_t> src; src.push_back(BOS);
        for (unsigned char c : text) src.push_back((int32_t)c);
        src.push_back(SEP);
        const size_t cap = src.size() + 3 * (size_t)max_windows + 2;
        fwd::KVCache cache(cfg_.n_layer, (size_t)cfg_.n_kv_head, (size_t)cfg_.head_dim, cap);
        std::vector<float> logits(cfg_.n_vocab, 0.f);
        fwd::transformer_forward(cfg_, *mw_, src, cache, logits.data(), *pool_);
        State st = State::Start;
        std::vector<int32_t> out; uint32_t wins = 0;
        for (size_t step = 0; step < 3 * (size_t)max_windows + 2; ++step) {
            int32_t best = -1; float bv = -std::numeric_limits<float>::infinity();
            for (int32_t t = 0; t < (int32_t)vocab; ++t)
                if (allowed(st, t) && logits[t] > bv) { bv = logits[t]; best = t; }            // argmax over the allowed tokens (ties: lowest id, like numpy / torch argmax)
            if (best < 0) throw std::runtime_error("goal translator: no allowed token (non-finite logits)");
            if (best == WIN && ++wins > max_windows) throw std::runtime_error("the plan reached the " + std::to_string(max_windows) + "-window cap without END");
            st = next(st, best);
            out.push_back(best);
            if (best == END) { if (last_logits) *last_logits = logits; return out; }
            std::vector<int32_t> one{best};
            fwd::transformer_forward(cfg_, *mw_, one, cache, logits.data(), *pool_);
        }
        throw std::runtime_error("the output was cut off before END");
    }

    std::string handle(const json::Value& req) const {
        try {
            const std::string op = req["op"].get_str("info");
            if (op == "info") return info();
            if (op == "goal_translate") {
                const auto toks = decode(req["text"].get_str(""));
                std::ostringstream o; o << "{\"ok\":true,\"windows\":[";
                bool first = true; size_t n = 0;
                for (size_t i = 0; i < toks.size(); ++i) {
                    if (toks[i] == WIN) {
                        if (!first) o << ",";
                        first = false; ++n;
                        if (toks[i + 1] == UNSPEC) o << "null";
                        else o << "[" << (toks[i + 1] - TURN_BASE) << "," << (toks[i + 2] - SPEED_BASE) << "]";
                    }
                }
                o << "],\"count\":" << n << ",\"tokens\":[";
                for (size_t i = 0; i < toks.size(); ++i) o << (i ? "," : "") << toks[i];
                o << "]}";
                return o.str();
            }
            if (op == "goal_logits") {
                need();
                const auto& a = req["tokens"];
                if (!a.is_array() || a.size() == 0) throw std::runtime_error("goal_logits: \"tokens\" must be a non-empty array of token ids");
                std::vector<int32_t> t;
                for (size_t i = 0; i < a.size(); ++i) {
                    long v = (long)a[i].get_int64(-1);
                    if (v < 0 || v >= (long)vocab) throw std::runtime_error("goal_logits: token id out of range");
                    t.push_back((int32_t)v);
                }
                fwd::KVCache cache(cfg_.n_layer, (size_t)cfg_.n_kv_head, (size_t)cfg_.head_dim, t.size() + 1);
                std::vector<float> logits(cfg_.n_vocab, 0.f);
                fwd::transformer_forward(cfg_, *mw_, t, cache, logits.data(), *pool_);
                std::ostringstream o; o.precision(9); o << "{\"ok\":true,\"logits\":[";
                for (size_t i = 0; i < logits.size(); ++i) o << (i ? "," : "") << logits[i];
                o << "]}";
                return o.str();
            }
            throw std::runtime_error("goal translator: unknown op '" + op + "' (info|goal_translate|goal_logits)");
        } catch (const std::exception& e) {
            std::string m = e.what(), esc;
            for (char c : m) { if (c == '"' || c == '\\') esc += '\\'; if ((unsigned char)c >= 0x20) esc += c; }
            return "{\"ok\":false,\"error\":\"" + esc + "\"}";
        }
    }

private:
    enum class State { Start, Win, Turn, After, Done };
    bool allowed(State s, int32_t t) const {
        switch (s) {
            case State::Start: return t == WIN;
            case State::Win:   return t == UNSPEC || (t >= TURN_BASE && t < SPEED_BASE);
            case State::Turn:  return t >= SPEED_BASE && t < SPEED_BASE + (int32_t)n_speed_bins;
            case State::After: return t == WIN || t == END;
            default: return false;
        }
    }
    State next(State, int32_t t) const {
        if (t == WIN) return State::Win;
        if (t == UNSPEC) return State::After;
        if (t >= TURN_BASE && t < SPEED_BASE) return State::Turn;
        if (t >= SPEED_BASE) return State::After;
        return State::Done;
    }
    void need() const { if (!loaded_) throw std::runtime_error("goal translator: no model loaded"); }

    bool loaded_ = false;
    util::ThreadPool* pool_ = nullptr;
    std::unique_ptr<loader::GGUFLoader> file_;
    std::unique_ptr<fwd::ModelWeights> mw_;
    engine::Config cfg_;
};

}  // namespace goaltr
