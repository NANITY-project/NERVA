// mini_json.hpp -- small strict JSON parser (for Hyprland's `j/` replies). Depth-limited, rejects trailing garbage.
#pragma once
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace nerva {

struct Json {
    enum Type { Null, Bool, Num, Str, Arr, Obj } t = Null;
    double n = 0; bool b = false; std::string s;
    std::vector<Json> a;                                           // array items, or object values
    std::vector<std::string> keys;                                 // object keys (parallel to `a`)

    const Json* get(const char* k) const {
        if (t != Obj) return nullptr;
        for (size_t i = 0; i < keys.size(); ++i) if (keys[i] == k) return &a[i];
        return nullptr;
    }
    double num(const char* k, double def = 0) const { const Json* j = get(k); return j && j->t == Num ? j->n : (j && j->t == Bool ? (j->b ? 1 : 0) : def); }
    std::string str(const char* k, const std::string& def = "") const { const Json* j = get(k); return j && j->t == Str ? j->s : def; }
    bool boolean(const char* k, bool def = false) const { const Json* j = get(k); return j ? (j->t == Bool ? j->b : (j->t == Num ? j->n != 0 : def)) : def; }
};

namespace json_detail {
struct P {
    const std::string& s; size_t i = 0; std::string err; int max_depth;
    bool fail(const char* m) { if (err.empty()) err = std::string(m) + " at byte " + std::to_string(i); return false; }
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i; }
    static void utf8(std::string& o, unsigned cp) {
        if (cp < 0x80) o += (char)cp;
        else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 63)); }
        else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
        else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 63)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
    }
    bool hex4(unsigned& v) {
        if (i + 4 > s.size()) return fail("short \\u escape");
        v = 0;
        for (int k = 0; k < 4; ++k) { char c = s[i++]; v <<= 4; if (c >= '0' && c <= '9') v |= c - '0'; else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10; else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10; else return fail("bad \\u escape"); }
        return true;
    }
    bool string(std::string& o) {
        ++i;                                                       // opening quote
        while (i < s.size()) {
            unsigned char c = (unsigned char)s[i++];
            if (c == '"') return true;
            if (c < 0x20) return fail("control character in string");
            if (c != '\\') { o += (char)c; continue; }
            if (i >= s.size()) break;
            char e = s[i++];
            switch (e) {
                case '"': o += '"'; break; case '\\': o += '\\'; break; case '/': o += '/'; break;
                case 'b': o += '\b'; break; case 'f': o += '\f'; break; case 'n': o += '\n'; break; case 'r': o += '\r'; break; case 't': o += '\t'; break;
                case 'u': {
                    unsigned v = 0; if (!hex4(v)) return false;
                    if (v >= 0xD800 && v < 0xDC00) {              // surrogate pair
                        if (i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') { i += 2; unsigned lo = 0; if (!hex4(lo)) return false; if (lo >= 0xDC00 && lo < 0xE000) v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00); else v = 0xFFFD; }
                        else v = 0xFFFD;
                    } else if (v >= 0xDC00 && v < 0xE000) v = 0xFFFD;
                    utf8(o, v); break;
                }
                default: return fail("bad escape");
            }
        }
        return fail("unterminated string");
    }
    bool value(Json& o, int depth) {
        if (depth > max_depth) return fail("nesting too deep");
        ws();
        if (i >= s.size()) return fail("unexpected end");
        char c = s[i];
        if (c == '{') {
            o.t = Json::Obj; ++i; ws();
            if (i < s.size() && s[i] == '}') { ++i; return true; }
            for (;;) {
                ws(); if (i >= s.size() || s[i] != '"') return fail("expected key");
                std::string k; if (!string(k)) return false;
                ws(); if (i >= s.size() || s[i] != ':') return fail("expected ':'"); ++i;
                Json v; if (!value(v, depth + 1)) return false;
                o.keys.push_back(std::move(k)); o.a.push_back(std::move(v));
                ws(); if (i >= s.size()) return fail("unexpected end");
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == '}') { ++i; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            o.t = Json::Arr; ++i; ws();
            if (i < s.size() && s[i] == ']') { ++i; return true; }
            for (;;) {
                Json v; if (!value(v, depth + 1)) return false;
                o.a.push_back(std::move(v));
                ws(); if (i >= s.size()) return fail("unexpected end");
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == ']') { ++i; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') { o.t = Json::Str; return string(o.s); }
        if (s.compare(i, 4, "true") == 0) { o.t = Json::Bool; o.b = true; i += 4; return true; }
        if (s.compare(i, 5, "false") == 0) { o.t = Json::Bool; o.b = false; i += 5; return true; }
        if (s.compare(i, 4, "null") == 0) { o.t = Json::Null; i += 4; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            size_t j = i; if (s[j] == '-') ++j;
            size_t d0 = j; while (j < s.size() && s[j] >= '0' && s[j] <= '9') ++j;
            if (j == d0) return fail("bad number");
            if (j < s.size() && s[j] == '.') { ++j; size_t f0 = j; while (j < s.size() && s[j] >= '0' && s[j] <= '9') ++j; if (j == f0) return fail("bad number"); }
            if (j < s.size() && (s[j] == 'e' || s[j] == 'E')) { ++j; if (j < s.size() && (s[j] == '+' || s[j] == '-')) ++j; size_t e0 = j; while (j < s.size() && s[j] >= '0' && s[j] <= '9') ++j; if (j == e0) return fail("bad number"); }
            std::string tok = s.substr(i, j - i); o.t = Json::Num; o.n = std::strtod(tok.c_str(), nullptr);
            if (!std::isfinite(o.n)) o.n = 0;
            i = j; return true;
        }
        return fail("unexpected character");
    }
};
}  // namespace json_detail

inline bool json_parse(const std::string& text, Json& out, std::string* err = nullptr, int max_depth = 48) {
    json_detail::P p{text, 0, "", max_depth};
    Json v;
    bool ok = p.value(v, 0);
    if (ok) { p.ws(); if (p.i != text.size()) { ok = false; p.fail("trailing characters"); } }
    if (ok) out = std::move(v); else if (err) *err = p.err;
    return ok;
}

}  // namespace nerva
