// Copyright 2026 CipherFlow (Shenzhen) Co., Ltd.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Sparse-packing bootstrap end-to-end example. Mirrors
// 5_ckks_regular_bootstrapping_v2.cu but uses BootstrappingConfigV2 with
// log_slots < LogN-1, sparsely encodes the input (real data in first
// 2^log_slots slots, zeros elsewhere), and reuses regular_bootstrapping_v2 —
// the existing API dispatches to the doubled-mode CtS fuse + single-EvalMod
// path automatically once context.set_slot_count(1 << log_slots) puts the
// encoder in sparse (gap_ > 1) mode.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <heongpu/heongpu.hpp>

#include "../example_util.h"

namespace
{

// Modulus chain sized for sparse bootstrap on N = 2^16. The pipeline assigns
// distinct roles to specific bands of the chain:
// [Q0 | ciphertext primes | StC primes | Sine double-angle primes |
//  Sine primes | CtS primes]. Piece counts (CtoS, StoC) determine how many
// CtS/StC primes are present, and the level_start fields on the
// EncodingMatrixConfig / EvalModConfig must point into the correct band.
//
// Primes are a prefix of example 5's proven set. generate_primes() can't
// substitute here: Q0 (61 bit) and the P primes (61 bit) exceed its
// MAX_USER_DEFINED_MOD_BIT_COUNT = 60 cap (include/kernel/defines.h).
struct SparseChain
{
    std::vector<uint64_t> Q;
    std::vector<uint64_t> P;
    int cts_level_start;
    int eval_mod_level_start;
    int stc_level_start;
    int cts_piece;
    int stc_piece;
};

SparseChain build_sparse_chain_n16(int log_slots)
{
    if (log_slots < 2 || log_slots > 14)
    {
        throw std::invalid_argument(
            "build_sparse_chain_n16: log_slots must be in [2, 14] at log_N=16. "
            "log_slots = 15 is fully packed (slot_count = N/2 = 32768) — use "
            "the regular bootstrap chain instead. "
            "log_slots > 15 is out of CKKS range at log_N=16 (max slots = N/2). "
            "log_slots <= 1 is currently unsupported: at log_num_slots_=1 the "
            "V-matrix has only 2 diagonals and one of the BSGS DeviceVectors "
            "(V_matrixs_index_[i] / V_mul / temp_result) collapses to size 0, "
            "causing E_diagonal_matrix_mult_kernel_cf to deref NULL. Tracked "
            "as a follow-up alongside log_slots=0 single-slot support.");
    }

    // Default CtS/StC BSGS piece counts (4, 3) clamped to the valid range
    // for sparse bootstrap: log_slots >= 2 * max(cts, stc).
    constexpr int kDefaultCtsPiece = 4;
    constexpr int kDefaultStcPiece = 3;
    const int cts_piece =
        std::max(1, std::min(kDefaultCtsPiece, log_slots / 2));
    const int stc_piece =
        std::max(1, std::min(kDefaultStcPiece, log_slots / 2));

    static constexpr uint64_t kQ0 = 0x10000000006e0001ULL;
    static constexpr uint64_t kCtPrimes[] = {
        0x10000140001ULL, 0xffffe80001ULL,   0xffffc40001ULL,
        0x100003e0001ULL, 0xffffb20001ULL,   0x10000500001ULL,
        0xffff940001ULL,  0xffff8a0001ULL,   0xffff820001ULL};
    static constexpr uint64_t kStcPrimes[] = {
        0x7fffe60001ULL, 0x7fffe40001ULL, 0x7fffe00001ULL};
    static constexpr uint64_t kSineDaPrimes[] = {
        0xfffffffff840001ULL, 0x1000000000860001ULL,
        0xfffffffff6a0001ULL};
    static constexpr uint64_t kSinePrimes[] = {
        0x1000000000980001ULL, 0xfffffffff5a0001ULL,
        0x1000000000b00001ULL, 0x1000000000ce0001ULL,
        0xfffffffff2a0001ULL};
    static constexpr uint64_t kCtsPrimes[] = {
        0x100000000060001ULL, 0xfffffffff00001ULL,
        0xffffffffd80001ULL, 0x1000000002a0001ULL};
    static constexpr uint64_t kPPrimes[] = {
        0x1fffffffffe00001ULL, 0x1fffffffffc80001ULL,
        0x1fffffffffb40001ULL, 0x1fffffffff500001ULL,
        0x1fffffffff420001ULL};

    constexpr int ct_count = sizeof(kCtPrimes) / sizeof(kCtPrimes[0]);
    constexpr int sine_da_count =
        sizeof(kSineDaPrimes) / sizeof(kSineDaPrimes[0]);
    constexpr int sine_count = sizeof(kSinePrimes) / sizeof(kSinePrimes[0]);

    SparseChain chain;
    chain.Q.reserve(1 + ct_count + stc_piece + sine_da_count + sine_count +
                    cts_piece);
    chain.Q.push_back(kQ0);
    for (int i = 0; i < ct_count; i++) chain.Q.push_back(kCtPrimes[i]);
    for (int i = 0; i < stc_piece; i++) chain.Q.push_back(kStcPrimes[i]);
    for (int i = 0; i < sine_da_count; i++) chain.Q.push_back(kSineDaPrimes[i]);
    for (int i = 0; i < sine_count; i++) chain.Q.push_back(kSinePrimes[i]);
    for (int i = 0; i < cts_piece; i++) chain.Q.push_back(kCtsPrimes[i]);
    chain.P.assign(std::begin(kPPrimes), std::end(kPPrimes));

    const int stc_start = 1 + ct_count;
    const int sine_da_start = stc_start + stc_piece;
    const int sine_start = sine_da_start + sine_da_count;
    const int cts_start = sine_start + sine_count;

    chain.cts_level_start = cts_start + cts_piece - 1;
    chain.eval_mod_level_start = sine_start + sine_count - 1;
    chain.stc_level_start = stc_start + stc_piece - 1;
    chain.cts_piece = cts_piece;
    chain.stc_piece = stc_piece;
    return chain;
}

} // namespace

int main(int argc, char* argv[])
{
    cudaSetDevice(0);

    int log_slots = 8;  // default: 256 active slots
    if (argc > 1)
    {
        log_slots = std::atoi(argv[1]);
    }

    heongpu::HEContext<heongpu::Scheme::CKKS> context =
        heongpu::GenHEContext<heongpu::Scheme::CKKS>(
            heongpu::sec_level_type::none);
    size_t poly_modulus_degree = 1 << 16;
    context->set_poly_modulus_degree(poly_modulus_degree);

    SparseChain chain = build_sparse_chain_n16(log_slots);
    std::cout << "log_slots=" << log_slots
              << "  CtoS_piece=" << chain.cts_piece
              << "  StoC_piece=" << chain.stc_piece
              << "  Q_len=" << chain.Q.size()
              << "  cts_L=" << chain.cts_level_start
              << "  eval_mod_L=" << chain.eval_mod_level_start
              << "  stc_L=" << chain.stc_level_start << "\n";
    // @company CipherFlow: enable sparse mode so encoder produces sparse-NTT'd
    // plaintexts (small FFT/NTT then expand to full N) and regular_bootstrapping_v2
    // takes the doubled-mode CtS fuse path (single EvalMod).
    context->set_slot_count(1 << log_slots);

    context->set_coeff_modulus_values(chain.Q, chain.P);

    context->generate();
    context->print_parameters();

    int h = 192;
    int ephemeral_secret_weight = 32;
    double scale = pow(2.0, 40);

    heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
    heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context, h);
    keygen.generate_secret_key_v2(secret_key);

    heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::Relinkey<heongpu::Scheme::CKKS> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    heongpu::Switchkey<heongpu::Scheme::CKKS>* swk_dense_to_sparse = nullptr;
    heongpu::Switchkey<heongpu::Scheme::CKKS>* swk_sparse_to_dense = nullptr;

    if (ephemeral_secret_weight > 0)
    {
        heongpu::Secretkey<heongpu::Scheme::CKKS> sparse_secret_key(
            context, ephemeral_secret_weight);
        keygen.generate_secret_key_v2(sparse_secret_key);

        swk_dense_to_sparse =
            new heongpu::Switchkey<heongpu::Scheme::CKKS>(context);
        keygen.generate_switch_key(*swk_dense_to_sparse, sparse_secret_key,
                                   secret_key);

        swk_sparse_to_dense =
            new heongpu::Switchkey<heongpu::Scheme::CKKS>(context);
        keygen.generate_switch_key(*swk_sparse_to_dense, secret_key,
                                   sparse_secret_key);
    }

    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
    heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context, public_key);
    heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators(context,
                                                                   encoder);

    const int active_slots = 1 << log_slots;
    const int log_n = static_cast<int>(log2(static_cast<double>(poly_modulus_degree)));
    const int gap = 1 << (log_n - 1 - log_slots);

    std::vector<Complex64> sparse_message(active_slots, Complex64(0.2, 0.4));

    heongpu::Plaintext<heongpu::Scheme::CKKS> P1(context);
    encoder.encode(P1, sparse_message, scale);

    heongpu::Ciphertext<heongpu::Scheme::CKKS> C1(context);
    encryptor.encrypt(C1, P1);

    // @company CipherFlow: regular_bootstrapping_v2 needs Q and scaling_factor
    // populated (the EvalMod precomputed weights depend on them); the 1-arg
    // ctor leaves these at zero which breaks precision.
    heongpu::EvalModConfig eval_mod_config(context->get_key_modulus()[0].value,
                                           chain.eval_mod_level_start, 256.0,
                                           16, 30, 3, 0, pow(2.0, 60));

    heongpu::BootstrappingConfigV2 boot_config(
        heongpu::EncodingMatrixConfig(
            heongpu::LinearTransformType::SLOTS_TO_COEFFS,
            chain.stc_level_start, /*bsgs_ratio=*/2.0f, chain.stc_piece),
        eval_mod_config,
        heongpu::EncodingMatrixConfig(
            heongpu::LinearTransformType::COEFFS_TO_SLOTS,
            chain.cts_level_start, /*bsgs_ratio=*/2.0f, chain.cts_piece));

    operators.generate_bootstrapping_params_v2(scale, boot_config);

    // Galois key must cover the matrix-mul rotations AND the Trace rotations
    // {2^i : i ∈ [log_slots, LogN-2]}.
    std::vector<int> key_index = operators.bootstrapping_key_indexs();
    for (int i = log_slots; i < log_n - 1; i++)
    {
        key_index.push_back(1 << i);
    }
    std::sort(key_index.begin(), key_index.end());
    key_index.erase(std::unique(key_index.begin(), key_index.end()),
                    key_index.end());
    std::cout << "Total galois key needed for sparse CKKS bootstrapping: "
              << key_index.size() << " (log_slots=" << log_slots << ")"
              << std::endl;
    heongpu::Galoiskey<heongpu::Scheme::CKKS> galois_key(context, key_index);
    keygen.generate_galois_key(galois_key, secret_key);

    // Drop all levels until one remains. Chain has (Q.size() - 1) drops
    // available — for log_slots >= 8 this is 24 (matches example 5), for
    // smaller log_slots it shrinks as CtS/StC bands shrink.
    const int drops = static_cast<int>(chain.Q.size()) - 1;
    for (int i = 0; i < drops; i++)
    {
        operators.mod_drop_inplace(C1);
    }

    std::cout << "Level before bootstrapping: " << C1.level() << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> cipher_boot =
        operators.regular_bootstrapping_v2(C1, galois_key, relin_key,
                                           swk_dense_to_sparse,
                                           swk_sparse_to_dense);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Level after bootstrapping: " << cipher_boot.level()
              << std::endl;
    std::cout << "Sparse bootstrapping time: " << milliseconds << " ms ("
              << milliseconds / 1000.0 << " seconds)" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    heongpu::Plaintext<heongpu::Scheme::CKKS> P_res1(context);
    decryptor.decrypt(P_res1, cipher_boot);
    std::vector<Complex64> decrypted_1;
    encoder.decode(decrypted_1, P_res1);

    // After set_slot_count + 3-arg encode/decode, decoded vector has exactly
    // slot_count_ = active_slots entries — slot j lives at decrypted_1[j], not
    // at decrypted_1[j*gap]. Compare directly.
    heongpu::PrecisionStats prec_stats =
        heongpu::get_precision_stats(sparse_message, decrypted_1);

    std::cout << "\n=== Sparse Bootstrapping Precision Statistics ===\n";
    std::cout << prec_stats.to_string() << std::endl;

    for (int j = 0; j < std::min(16, active_slots); j++)
    {
        std::cout << j << "-> EXPECTED:" << sparse_message[j]
                  << " - ACTUAL:" << decrypted_1[j] << std::endl;
    }
    std::cout << std::endl;

    delete swk_dense_to_sparse;
    delete swk_sparse_to_dense;
    return EXIT_SUCCESS;
}
