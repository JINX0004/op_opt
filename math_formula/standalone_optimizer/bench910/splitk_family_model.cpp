#include "mip_solver.h"

#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    // Exploratory 910B3 profile plus source-in-context mode-6 FP32 atomic
    // Fixpipe demand measured by msprof on device 4. This rate was fitted
    // using this experiment's wide cases, so the family decisions below are
    // diagnostic, not out-of-sample validation.
    mmopt::Hardware hardware{
        20, 512ULL * 1024 * 1024,
        {{"launch", {"count", 2.68, 1.0}},
         {"gm_read", {"bytes", 0.109620, 3710212.0}},
         {"gm_write", {"bytes", 0.162586, 5229502.0}},
         {"cube", {"flop", 0.0, 14719432.0}},
         {"vector_add", {"ops", 0.0, 1000000.0}},
         {"nd2nz", {"elements", 0.0, 1000000.0}},
         {"sync", {"count", 0.1, 1.0}},
         {"split_partial", {"count", 1.28, 1.0}},
         {"sc_fp32_store", {"bytes", 0.2432, 34190.0}},
         {"sc_fp32_store_steady", {"bytes", 0.0, 91000.0}},
         // Mode-5 source-in-context FP32 demand and incremental IterateAll
         // scalar work from the 49152x384x1536 paired tile experiment.
         {"sc_mode5_fixpipe", {"bytes", 0.0, 108000.0}},
         {"sc_mode5_iterate", {"count", 1.8, 1.0}},
         {"sc_atomic_extra", {"bytes", 0.0541, 106618.0}},
         {"sc_cast", {"elements", 0.0156, 116718.0}},
         {"sc_gm_to_l1_a", {"bytes", 0.3124, 270810.0}},
         {"sc_gm_to_l1_b", {"bytes", 0.0227, 115414.0}}}};
    std::cout << std::fixed << std::setprecision(5);
    if (argc > 1 && std::string(argv[1]) == "source") {
        std::cout << "m,n,k,mode,source_geometry_model_us\n";
        std::uint64_t m, n, k, mode, cm, cn, ck, bm, bn, bk, sm, sn, cores;
        while (std::cin >> m >> n >> k >> mode >> cm >> cn >> ck >> bm >> bn >> bk >> sm >> sn >> cores) {
            if ((mode != 2 && mode != 3 && mode != 5 && mode != 6) ||
                bk == 0 || ck % bk != 0) {
                std::cout << m << ',' << n << ',' << k << ',' << mode
                          << ",error:invalid source mode or K step\n";
                continue;
            }
            mmopt::PlanDecision d;
            d.family = mode == 3 ? mmopt::KernelFamily::DeterministicSplitK :
                mode == 2 ? mmopt::KernelFamily::SingleCoreSplitKMode2 :
                mode == 5 ? mmopt::KernelFamily::SingleCoreSplitKMode5 :
                mmopt::KernelFamily::SingleCoreSplitKMode6;
            d.used_cores = cores;
            d.single_core_m = cm;
            d.single_core_n = cn;
            d.single_core_k = ck;
            d.base_m = bm;
            d.base_n = bn;
            d.base_k = bk;
            d.step_m = sm;
            d.step_n = sn;
            d.step_k = ck / bk;
            try {
                const double score = mmopt::ReducedObjectiveUs({m,n,k,"fp16"}, hardware, d);
                std::cout << m << ',' << n << ',' << k << ',' << mode << ',' << score << '\n';
            } catch (const std::exception& error) {
                std::cout << m << ',' << n << ',' << k << ',' << mode << ",error:"
                          << error.what() << '\n';
            }
        }
        return 0;
    }
    std::cout << "m,n,k,single_us,deterministic_us,selected,"
                 "single_mode,single_core_m,single_core_n,det_core_m,det_core_n\n";
    std::uint64_t m, n, k;
    while (std::cin >> m >> n >> k) {
        try {
            const auto result = mmopt::SolveReducedMip({m, n, k, "fp16"}, hardware);
            const mmopt::MipPlan* single = nullptr;
            for (const auto* plan : {&result.single_core_split_k_mode2,
                                     &result.single_core_split_k_mode5,
                                     &result.single_core_split_k_mode6}) {
                if (*plan && (!single || (*plan)->surrogate_us < single->surrogate_us))
                    single = &**plan;
            }
            const auto* det = result.deterministic_split_k
                ? &*result.deterministic_split_k : nullptr;
            const char* selected = !single ? "deterministic" :
                !det ? "single" :
                single->surrogate_us < det->surrogate_us ? "single" : "deterministic";
            std::cout << m << ',' << n << ',' << k << ',';
            if (single) std::cout << single->surrogate_us;
            std::cout << ',';
            if (det) std::cout << det->surrogate_us;
            std::cout << ',' << (single || det ? selected : "infeasible") << ',';
            if (single) {
                const auto f = single->decision.family;
                std::cout << (f == mmopt::KernelFamily::SingleCoreSplitKMode2 ? 2 :
                              f == mmopt::KernelFamily::SingleCoreSplitKMode5 ? 5 : 6)
                          << ',' << single->decision.single_core_m << ','
                          << single->decision.single_core_n;
            } else std::cout << ",,";
            std::cout << ',';
            if (det) std::cout << det->decision.single_core_m << ','
                               << det->decision.single_core_n;
            else std::cout << ',';
            std::cout << '\n';
        } catch (const std::exception& error) {
            std::cout << m << ',' << n << ',' << k << ",,,error:"
                      << error.what() << ",,,,,\n";
        }
    }
}
