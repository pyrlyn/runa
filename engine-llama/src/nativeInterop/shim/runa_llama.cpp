#include "runa_llama.h"

#include "chat.h"
#include "common.h"
#include "json.h"
#include "log.h"
#include "sampling.h"
#include "unicode.h"

#include <exception>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string last_error;

// An exception must not reach Kotlin: it cannot unwind through frames that are not C++.
template <typename T, typename F> T guarded(T failed, F && body) {
    try {
        last_error.clear();
        return body();
    } catch (const std::exception & e) {
        last_error = e.what();
    } catch (...) {
        last_error = "unknown error";
    }
    return failed;
}

// The common library logs to the terminal by itself. Its failures come back as errors instead.
void silence_common_log() { common_log_set_verbosity_thold(-1); }

std::string random_call_id() {
    static const char alphabet[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::random_device                    device;
    std::uniform_int_distribution<size_t> pick(0, sizeof(alphabet) - 2);
    std::string                           id = "call_";
    for (int i = 0; i < 24; i++) {
        id += alphabet[pick(device)];
    }
    return id;
}

common_json event(const char * kind, const common_json & value) {
    common_json result = common_json::object();
    result[kind]       = value;
    return result;
}

} // namespace

struct runa_chat {
    common_chat_templates_ptr templates;
};

struct runa_turn {
    const llama_vocab *   vocab = nullptr;
    common_sampler_ptr    sampler;
    bool                  chat = false;
    common_chat_session   session;
    std::set<llama_token> preserved;
    std::string           start;

    // Pieces that do not end a UTF-8 sequence yet: the parser takes whole characters.
    common_chat_input pending;

    // What the caller knows of the answer so far.
    common_chat_msg          message;
    std::vector<std::string> call_ids;
    std::set<size_t>         announced;

    std::string reported;

    // What `next` adds to `message`. The order of llama.cpp's own server (update_chat_msg): a call's
    // name can still grow while it is the last thing parsed, so it is held back until something
    // else follows it, or until the answer has `ended`.
    const char * report(common_chat_msg next, bool ended) {
        common_json events   = common_json::array();
        const auto  announce = [&](size_t index) {
            common_json call = common_json::object();
            call["index"]    = index;
            call["id"]       = message.tool_calls[index].id;
            call["name"]     = message.tool_calls[index].name;
            events.push_back(event("call", call));
            announced.insert(index);
        };
        if (!next.empty()) {
            next.set_tool_call_ids(call_ids, random_call_id);
            const auto diffs = common_chat_msg_diff::compute_diffs(message, next);
            message          = std::move(next);
            for (const auto & diff : diffs) {
                const std::string & arguments = diff.tool_call_delta.arguments;
                for (size_t index = 0; index < message.tool_calls.size(); index++) {
                    if (announced.count(index) == 0 && !message.tool_calls[index].name.empty() &&
                        (diff.tool_call_index != index || !arguments.empty())) {
                        announce(index);
                    }
                }
                if (!diff.content_delta.empty()) {
                    events.push_back(event("content", diff.content_delta));
                }
                if (diff.tool_call_index != std::string::npos && !arguments.empty() &&
                    announced.count(diff.tool_call_index) > 0) {
                    common_json added = common_json::object();
                    added["index"]    = diff.tool_call_index;
                    added["text"]     = arguments;
                    events.push_back(event("arguments", added));
                }
            }
        }
        if (ended) {
            for (size_t index = 0; index < message.tool_calls.size(); index++) {
                if (announced.count(index) == 0 && !message.tool_calls[index].name.empty()) {
                    announce(index);
                }
            }
        }
        // A model can write bytes that are not UTF-8; they must not end the answer.
        reported = events.dump_safe();
        return reported.c_str();
    }
};

const char * runa_last_error(void) { return last_error.c_str(); }

struct runa_chat * runa_chat_init(const struct llama_model * model, const char * source) {
    return guarded<runa_chat *>(nullptr, [&] {
        silence_common_log();
        auto chat       = std::make_unique<runa_chat>();
        chat->templates = common_chat_templates_init(model, source != nullptr ? source : "");
        return chat.release();
    });
}

void runa_chat_free(struct runa_chat * chat) { delete chat; }

struct runa_turn * runa_turn_init(const struct llama_model *   model,
                                  const struct runa_chat *     chat,
                                  const char *                 request,
                                  const struct runa_sampling * sampling) {
    return guarded<runa_turn *>(nullptr, [&] {
        silence_common_log();
        auto turn   = std::make_unique<runa_turn>();
        turn->vocab = llama_model_get_vocab(model);

        common_params_sampling params;
        params.temp           = sampling->temperature;
        params.top_p          = sampling->top_p;
        params.min_p          = sampling->min_p;
        params.penalty_repeat = sampling->repeat_penalty;
        params.top_k          = sampling->top_k;
        params.seed           = sampling->seed;
        params.no_perf        = true;

        if (chat != nullptr && request != nullptr) {
            const common_json body = common_json::parse(request);

            common_chat_templates_inputs inputs;
            inputs.messages    = common_chat_msgs_parse_oaicompat(body.at("messages"));
            inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(body.value("tool_choice", "auto"));
            if (body.contains("tools")) {
                inputs.tools = common_chat_tools_parse_oaicompat(body.at("tools"));
            }
            if (body.contains("json_schema")) {
                inputs.json_schema = body.at("json_schema").dump();
            }
            // As llama.cpp's server: several calls in one answer only when the template can show them.
            inputs.parallel_tool_calls =
                body.contains("parallel_tool_calls")
                    ? body.at("parallel_tool_calls").get<bool>()
                    : common_chat_templates_get_caps(chat->templates.get()).at("supports_parallel_tool_calls").get<bool>();

            turn->session = common_chat_session(chat->templates.get(), turn->vocab, inputs);
            turn->session.apply_sampling(params);
            turn->preserved = params.preserved_tokens;
            turn->chat      = true;

            common_json start = common_json::object();
            start["prompt"]   = turn->session.prompt().str();
            start["stops"]    = turn->session.additional_stops();
            turn->start       = start.dump_safe();
        }

        turn->sampler.reset(common_sampler_init(model, params));
        if (!turn->sampler) {
            throw std::runtime_error("llama.cpp cannot build the sampler");
        }
        return turn.release();
    });
}

void runa_turn_free(struct runa_turn * turn) { delete turn; }

const char * runa_turn_start(const struct runa_turn * turn) { return turn->chat ? turn->start.c_str() : nullptr; }

llama_token runa_turn_sample(struct runa_turn * turn, struct llama_context * context) {
    return guarded<llama_token>(LLAMA_TOKEN_NULL, [&] {
        const llama_token token = common_sampler_sample(turn->sampler.get(), context, -1);
        common_sampler_accept(turn->sampler.get(), token, true);
        return token;
    });
}

const char * runa_turn_accept(struct runa_turn * turn, llama_token token) {
    return guarded<const char *>(nullptr, [&] {
        // A token the grammar or the parser looks for is written out; any other control token has no text.
        const bool special = turn->preserved.count(token) > 0;
        turn->pending.append(common_token_to_piece(turn->vocab, token, special), token);
        if (!common_utf8_is_complete(turn->pending.text)) {
            return "[]";
        }
        common_chat_input chunk;
        std::swap(chunk, turn->pending);
        return turn->report(turn->session.feed(chunk), false);
    });
}

const char * runa_turn_finish(struct runa_turn * turn, bool complete) {
    return guarded<const char *>(nullptr, [&] {
        common_chat_input chunk;
        std::swap(chunk, turn->pending);
        // finish() rejects an answer that stops in the middle of what its format allows.
        return turn->report(complete ? turn->session.finish(chunk) : turn->session.feed(chunk), true);
    });
}
