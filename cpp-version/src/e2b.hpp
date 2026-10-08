#pragma once
#include <string>
#include <torch/torch.h>

/**
 * @brief E2b numeric conformance runner (thesis §8.8, §9.6).
 *
 * Loads the framework-neutral weight file shared by all five implementations,
 * runs the fixed probe forward pass, and writes the decoder logits as
 * little-endian float32 so that the numeric paths can be compared directly.
 *
 * This deliberately does not read translator.pth and does not need a trained
 * checkpoint of its own: the whole point of the shared file is that an
 * implementation without trained weights can still participate in the
 * comparison.
 *
 * @param dir  Directory holding arch.txt, manifest.tsv, probe.txt, model.bin.
 * @param tag  Output tag; writes logits_<tag>.bin into dir.
 * @return 0 on success.
 */
int run_e2b(const std::string& dir, const std::string& tag);

/**
 * @brief E2c gradient conformance runner (thesis §8.9, §9.7).
 *
 * Loads the same shared weight file as run_e2b, then runs one *training* step
 * on the fixed probe -- teacher forcing, cross-entropy with label smoothing
 * 0.1 and padding index 0 ignored, exactly as src/training.cpp does it --
 * calls backward(), and writes every parameter gradient as little-endian
 * float32 in sorted-parameter-name order.
 *
 * E2b verifies the forward pass; this verifies the backward one. The
 * distinction matters because §9.2's quality difference between independently
 * trained implementations is unattributable: if the gradients agree, the five
 * were descending the same surface and the divergence is down to
 * initialisation and data order alone.
 *
 * Gradients are captured before clip_grad_norm_, which projects onto a ball of
 * radius 1.0 and would mask a disagreement in magnitude.
 *
 * @param dir  Directory holding arch.txt, manifest.tsv, probe.txt, model.bin.
 * @param tag  Output tag; writes grads_<tag>.bin and grads_<tag>.txt into dir.
 * @return 0 on success.
 */
int run_e2c(const std::string& dir, const std::string& tag);
