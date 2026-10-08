#pragma once
#include <string>
#include "model/translator.hpp"
#include "tokenizer.hpp"

/**
 * @brief The decoding policy actually in force.
 *
 * One definition, used by the decoder and read by the E3 conformance check.
 * Previously the check printed these five values as a string literal while the
 * decoder took its own defaults, so the two could disagree without anything
 * failing -- the check could not detect a divergence in the policy it claimed
 * to verify (thesis 9.5). Changing a value here now changes both what runs and
 * what is reported.
 */
struct DecodingPolicy {
    int beam = 3;
    float lp = 0.6f;
    float rep_penalty = 1.4f;
    int max_steps = 30;
    float temperature = 0.8f;
};

/** The single policy instance the decoder uses. */
const DecodingPolicy& decoding_policy();

/**
 * @brief Translate an English sentence to Bengali using beam search.
 * 
 * @param m Trained Translator model
 * @param en_tok English tokenizer
 * @param bn_tok Bengali tokenizer
 * @param sentence Input English sentence
 * @param beam_size Number of beams to maintain (default 3)
 * @param lp Length penalty exponent (default 0.6)
 * @param rep_penalty Repetition penalty factor (default 1.4)
 *        - First occurrence: divided/multiplied by 1.4
 *        - Subsequent occurrences: divided/multiplied by 1.4 * 1.5
 * @return Translated Bengali sentence
 */
std::string translate_sentence(Translator& m,
                               SPTokenizer& en_tok,
                               SPTokenizer& bn_tok,
                               const std::string& sentence,
                               int beam_size = decoding_policy().beam,
                               float lp = decoding_policy().lp,
                               float rep_penalty = decoding_policy().rep_penalty,
                               bool trace = false);
