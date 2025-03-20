//#include <iostream>
//#include <vector>
//#include <algorithm>
//#include <cmath>
//#include <sstream>
//#include <fstream>
//#include <random>
//#include "vec.h"
//
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//
//#include "stb_image_write.h"
//
//const int resolution = 128;
//
////基于 Hammersley 序列的低差异序列生成函数,用于生成辅助重要性采样
//Vec2f Hammersley(uint32_t i, uint32_t N) { // 0-1
//    uint32_t bits = (i << 16u) | (i >> 16u);
//    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
//    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
//    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
//    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
//    float rdi = float(bits) * 2.3283064365386963e-10;
//    return {float(i) / float(N), rdi};
//}
//
//Vec3f ImportanceSampleGGX(Vec2f Xi, Vec3f N, float roughness) {
//    float a = roughness * roughness;
//    //TODO: in spherical space - Bonus 1
//    float theta = atan(a * sqrt(Xi.x) / sqrt(1.0f - Xi.x));
//    float phi = 2.0 * PI * Xi.y;
//
//
//    //TODO: from spherical space to cartesian space - Bonus 1
//    float sinTheta = sin(theta);
//    float cosTheta = cos(theta);
//    Vec3f H = Vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
//
//    //TODO: tangent coordinates - Bonus 1
//    Vec3f up = abs(N.z) < 0.999 ? Vec3f(0.0, 0.0, 1.0) : Vec3f(1.0, 0.0, 0.0);
//    Vec3f tangent = normalize(cross(up, N));
//    Vec3f bitangent = cross(N, tangent);
//
//    //TODO: transform H to tangent space - Bonus 1
//    Vec3f sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
//    return normalize(sampleVec);
//}
//
//float GeometrySchlickGGX(float NdotV, float roughness) {
//    // TODO: To calculate Schlick G1 here - Bonus 1
//    float a = roughness;
//    float k = (a * a) / 2.0f;
//
//    float nom = NdotV;
//    float denom = NdotV * (1.0f - k) + k;
//
//    return nom / denom;
//}
//
//float GeometrySmith(float roughness, float NoV, float NoL) {
//    float ggx2 = GeometrySchlickGGX(NoV, roughness);
//    float ggx1 = GeometrySchlickGGX(NoL, roughness);
//
//    return ggx1 * ggx2;
//}
//
//
//
//Vec3f IntegrateBRDF(Vec3f V, float roughness) {
//    const int sample_count = 1024;
//    float A = 0.0;
//    float B = 0.0;
//    Vec3f N = Vec3f(0.0, 0.0, 1.0);
//    for (int i = 0; i < sample_count; i++) {
//        //值范围在0-1
//        Vec2f Xi = Hammersley(i, sample_count);
//        //重要性采样得到的法线
//        Vec3f H = ImportanceSampleGGX(Xi, N, roughness);
//        //通过重要性采样得到的法线计算光线方向
//        Vec3f L = normalize(H * 2.0f * dot(V, H) - V);
//
//        //n·i
//        float NoL = std::max(L.z, 0.0f);
//        //n·m
//        float NoH = std::max(H.z, 0.0f);
//        //o·m
//        float VoH = std::max(dot(V, H), 0.0f);
//        //o·n
//        float NoV = std::max(dot(N, V), 0.0f);
//
//
//        // TODO: To calculate (fr * ni) / p_o here - Bonus 1
//        float Fc = pow(1.0f - VoH, 5.0f);
//        float G = GeometrySmith(roughness, NoV, NoL);
//
//        //weight为重要性采样得到的 brdf/pdf的最终化简结果
//        float weight  = VoH * G / (NoV * NoH);
//
//        //no split sum 
//        A += weight;
//
//        // // Split Sum - Bonus 2
//        // A += (1.0 - Fc) * weight;
//        // B += Fc * weight;
//    }
//    return { A / sample_count, A / sample_count, A / sample_count }; // No split sum version
//    // return { A / sample_count, B / sample_count, 0.0 };  // Split sum
//}
//
//int main() {
//    uint8_t data[resolution * resolution * 3];
//    float step = 1.0 / resolution;
//    for (int i = 0; i < resolution; i++) {
//        float roughness = step * (static_cast<float>(i) + 0.5f);
//        for (int j = 0; j < resolution; j++) {
//            float NdotV = step * (static_cast<float>(j) + 0.5f);
//            Vec3f V = Vec3f(std::sqrt(1.f - NdotV * NdotV), 0.f, NdotV);
//
//            Vec3f irr = IntegrateBRDF(V, roughness);
//
//            data[(i * resolution + j) * 3 + 0] = uint8_t(irr.x * 255.0);
//            data[(i * resolution + j) * 3 + 1] = uint8_t(irr.y * 255.0);
//            data[(i * resolution + j) * 3 + 2] = uint8_t(irr.z * 255.0);
//        }
//    }
//    stbi_flip_vertically_on_write(true);
//    stbi_write_png("Emu_IS_LUT.png", resolution, resolution, 3, data, resolution * 3);
//    
//    std::cout << "Finished precomputed!" << std::endl;
//    return 0;
//}