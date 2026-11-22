// A C API over llama.cpp's `common` library, see CMakeLists.txt.
//
// No function here throws. One that can fail returns NULL, or LLAMA_TOKEN_NULL, and
// runa_last_error() then says why. The strings these functions return belong to the object they
// were asked of and last until its next call. Like llama.cpp itself, nothing here may be entered
// twice at once for the same model.
#ifndef RUNA_LLAMA_H
#define RUNA_LLAMA_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// The chat templates of a model.
struct runa_chat;

// One answer of a model: its sampler and, for a chat, its prompt and the parser of what the model writes.
struct runa_turn;

struct runa_sampling {
    float    temperature;
    float    top_p;
    float    min_p;
    float    repeat_penalty;
    int32_t  top_k;
    uint32_t seed;
};

// Why the last call on this thread failed.
const char * runa_last_error(void);

// `source` replaces the model's own template: a Jinja template or the name of a built-in one,
// such as "chatml". NULL takes the model's, and ChatML when it has none.
struct runa_chat * runa_chat_init(const struct llama_model * model, const char * source);
void               runa_chat_free(struct runa_chat * chat);

// `request` is a JSON object in the words of OpenAI's chat completions: `messages`, and optionally
// `tools`, `tool_choice` ("auto", "required" or "none"), `parallel_tool_calls` and `json_schema`.
// Without `chat` and `request` the turn only samples: the caller has its own prompt.
struct runa_turn * runa_turn_init(const struct llama_model * model,
                                  const struct runa_chat *   chat,
                                  const char *               request,
                                  const struct runa_sampling * sampling);
void               runa_turn_free(struct runa_turn * turn);

// A chat turn's `{"prompt": ..., "stops": [...]}`: the rendered chat and the strings that end the
// answer besides the model's own end token.
const char * runa_turn_start(const struct runa_turn * turn);

// The next token after the last one `context` evaluated.
llama_token runa_turn_sample(struct runa_turn * turn, struct llama_context * context);

// What `token` adds to a chat turn's answer, as a JSON list of `{"content": text}`,
// `{"call": {"index", "id", "name"}}` and `{"arguments": {"index", "text"}}`. A call is announced
// once, with its whole name, before any of its arguments. `token` must not be an end token.
const char * runa_turn_accept(struct runa_turn * turn, llama_token token);

// The same for what was held back until the answer ended. Nothing may be accepted after it.
// `complete` says the model ended the answer itself, so it must be whole: a tool call it left
// unfinished is an error then. An answer that was cut short is reported as far as it got.
const char * runa_turn_finish(struct runa_turn * turn, bool complete);

#ifdef __cplusplus
}
#endif

#endif // RUNA_LLAMA_H
