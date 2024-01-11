#include <iostream>
#include <vector>
#include <array>
#include "LuminanceColormap.hpp"

LuminanceColormap::LuminanceColormap(const std::vector<std::array<double, 3>>& colors, int shades_for_each_color)
: shades_for_each_color(shades_for_each_color) {
    generateColormap(colors);
}

void LuminanceColormap::generateColormap(const std::vector<std::array<double, 3>>& colors) {
    for (const auto& color : colors) {
        std::vector<std::array<double, 3>> shades;
        for (int j = 0; j < shades_for_each_color; ++j) {
            double shade_factor = static_cast<double>(j) / shades_for_each_color;
            std::array<double, 3> shaded_color = {
                color[0] * (1 - shade_factor),
                color[1] * (1 - shade_factor),
                color[2] * (1 - shade_factor)
            };
            shades.push_back(shaded_color);
        }
        colormap.push_back(shades);
    }
}

std::array<double, 3> LuminanceColormap::getColor(int colorIdx, int shadeIdx) const {
    // Adjust colorIdx to ensure it's within the bounds
    if (colorIdx < 0) {
        colorIdx = 0;
    } else if (colorIdx >= static_cast<int>(colormap.size())) {
        colorIdx = static_cast<int>(colormap.size()) - 1;
    }

    // Adjust shadeIdx similarly
    if (shadeIdx < 0) {
        shadeIdx = 0;
    } else if (shadeIdx >= shades_for_each_color) {
        shadeIdx = shades_for_each_color - 1;
    }

    return colormap[colorIdx][shadeIdx];
}

std::pair<int, int> LuminanceColormap::getShape() const {
    return {static_cast<int>(colormap.size()), shades_for_each_color};
}

std::vector<std::vector<std::array<double, 3>>> LuminanceColormap::getColormap() const {
    return colormap;
}
