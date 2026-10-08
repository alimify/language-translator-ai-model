/**Translator AI - English to Bengali Neural Machine Translation.*/
import { backendName } from "./tf.js";

const DEVICE = backendName;

import { Config } from "./config.js";
import { SPTokenizer } from "./tokenizer.js";
import { Translator } from "./model/translator.js";
import { get_raw_data } from "./data.js";
import { run_training } from "./training.js";
import { translate_sentence } from "./inference.js";

export {
    Config,
    SPTokenizer,
    Translator,
    get_raw_data,
    run_training,
    translate_sentence,
    DEVICE,
};
