// Persistent text-encoder process for job queues: load the LLM2Vec encoder
// once, then answer prompts over stdin/stdout until EOF.
//
// Protocol (little-endian):
//   request:  u32 n, then n bytes of UTF-8 prompt   (n == 0 or EOF: exit)
//   response: u32 0, then 4096 f32                  (success)
//             u32 1, u32 m, then m bytes of message  (failure; process continues)
// After loading, one line "ready backend=<name>" is written to stderr.
#include "llm_text_encoder.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

namespace {
bool read_exact(void *data, size_t n) { return std::fread(data, 1, n, stdin) == n; }
void write_u32(uint32_t value) { std::fwrite(&value, sizeof(value), 1, stdout); }
} // namespace

int main(int argc, char **argv) try {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " TEXT_MODEL.gguf|TEXT_BUNDLE_DIR\n";
        return 2;
    }
    auto encoder = kimodo::detail::llm_text_encoder::load(argv[1]);
    if (!encoder) {
        std::cerr << "load failed: " << encoder.error() << '\n';
        return 1;
    }
    std::cerr << "ready backend=" << (*encoder)->backend_name() << std::endl;
    std::string prompt;
    for (;;) {
        uint32_t n = 0;
        if (!read_exact(&n, sizeof(n)) || n == 0) return 0;
        if (n > (1U << 20)) {
            std::cerr << "request too large: " << n << '\n';
            return 1;
        }
        prompt.resize(n);
        if (!read_exact(prompt.data(), n)) return 1;
        auto embedding = (*encoder)->encode(prompt);
        if (embedding) {
            write_u32(0);
            std::fwrite(embedding->data(), sizeof(float), embedding->size(), stdout);
        } else {
            write_u32(1);
            write_u32(static_cast<uint32_t>(embedding.error().size()));
            std::fwrite(embedding.error().data(), 1, embedding.error().size(), stdout);
        }
        std::fflush(stdout);
    }
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
