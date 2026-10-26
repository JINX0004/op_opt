// Query the active device and installed CANN platform limits used by SolveBase.
#include <acl/acl.h>
#include <platform/platform_ascendc.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void Check(int code, const char* call) {
    if (code) throw std::runtime_error(std::string(call) + ": " + std::to_string(code));
}
}

int main(int argc, char** argv) {
    int device = argc > 1 ? std::stoi(argv[1]) : 0;
    try {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "aclrtSetDevice");
        const char* soc = aclrtGetSocName();
        if (!soc) throw std::runtime_error("aclrtGetSocName returned null");
        auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance(soc);
        if (!platform) throw std::runtime_error("PlatformAscendCManager returned null");
        std::cout << "{\"device\":" << device << ",\"soc\":\"" << soc << "\""
                  << ",\"aic\":" << platform->GetCoreNumAic()
                  << ",\"aiv\":" << platform->GetCoreNumAiv();
        const struct { const char* name; platform_ascendc::CoreMemType kind; } memories[] = {
            {"l1_bytes", platform_ascendc::CoreMemType::L1},
            {"l0a_bytes", platform_ascendc::CoreMemType::L0_A},
            {"l0b_bytes", platform_ascendc::CoreMemType::L0_B},
            {"l0c_bytes", platform_ascendc::CoreMemType::L0_C},
            {"ub_bytes", platform_ascendc::CoreMemType::UB},
            {"l2_bytes", platform_ascendc::CoreMemType::L2},
        };
        for (const auto& memory : memories) {
            std::uint64_t bytes = 0;
            platform->GetCoreMemSize(memory.kind, bytes);
            std::cout << ",\"" << memory.name << "\":" << bytes;
        }
        std::cout << "}\n";
        Check(aclrtResetDevice(device), "aclrtResetDevice");
        Check(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "hardware_probe: " << e.what() << '\n';
        return 1;
    }
}
