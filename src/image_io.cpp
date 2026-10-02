#include "image_io.h"

#include <algorithm>
#include <cmath>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

bool load_image(const std::string& path, Dataset& data) {
    int w = 0, h = 0, channels = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!pixels) return false;

    const int n = w * h;
    data.width = w;
    data.height = h;
    data.aos.resize(n);
    data.r.resize(n);
    data.g.resize(n);
    data.b.resize(n);
    for (int i = 0; i < n; ++i) {
        const float r = pixels[3 * i + 0], g = pixels[3 * i + 1], b = pixels[3 * i + 2];
        data.aos[i] = {r, g, b};
        data.r[i] = r;
        data.g[i] = g;
        data.b[i] = b;
    }
    stbi_image_free(pixels);
    return true;
}

bool save_segmentation(const std::string& path, const Dataset& data, const Result& result) {
    const int n = data.size();
    std::vector<unsigned char> out(3 * static_cast<size_t>(n));
    auto to_byte = [](float v) {
        return static_cast<unsigned char>(std::clamp(std::lround(v), 0L, 255L));
    };
    for (int i = 0; i < n; ++i) {
        const Pixel& c = result.centers[result.labels[i]];
        out[3 * i + 0] = to_byte(c.r);
        out[3 * i + 1] = to_byte(c.g);
        out[3 * i + 2] = to_byte(c.b);
    }
    return stbi_write_png(path.c_str(), data.width, data.height, 3, out.data(), data.width * 3) != 0;
}
