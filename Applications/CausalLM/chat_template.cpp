// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    chat_template.cpp
 * @date    10 Apr 2026
 * @brief   Hugging Face chat template adapter for OpenAI-style chat inputs.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jungwon-Lee <jungone.lee@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "chat_template.h"

#include <minja/chat-template.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace causallm {

namespace {

using OrderedJson = nlohmann::ordered_json;

bool fileExists(const std::string &path) {
  std::ifstream file(path);
  return file.good();
}

std::string readTextFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file: " + path);

  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

nlohmann::json readJsonFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file: " + path);

  nlohmann::json data;
  file >> data;
  return data;
}

OrderedJson toOrderedJson(const nlohmann::json &value) {
  if (value.is_object()) {
    OrderedJson out = OrderedJson::object();
    for (auto it = value.begin(); it != value.end(); ++it)
      out[it.key()] = toOrderedJson(it.value());
    return out;
  }

  if (value.is_array()) {
    OrderedJson out = OrderedJson::array();
    for (const auto &item : value)
      out.push_back(toOrderedJson(item));
    return out;
  }

  return OrderedJson::parse(value.dump());
}

std::string tokenContent(const nlohmann::json &value) {
  if (value.is_null())
    return "";

  if (value.is_string())
    return value.get<std::string>();

  if (value.is_object() && value.contains("content") &&
      value["content"].is_string())
    return value["content"].get<std::string>();

  return "";
}

std::string findToken(const nlohmann::json &tokenizer_config,
                      const nlohmann::json &special_tokens,
                      const std::string &name) {
  if (special_tokens.contains(name)) {
    std::string token = tokenContent(special_tokens[name]);
    if (!token.empty())
      return token;
  }

  if (tokenizer_config.contains(name))
    return tokenContent(tokenizer_config[name]);

  return "";
}

bool hasArrayItems(const nlohmann::json &value, const std::string &key) {
  return value.is_object() && value.contains(key) && value[key].is_array() &&
         !value[key].empty();
}

std::string textFromContentParts(const nlohmann::json &content) {
  std::string text;
  for (const auto &part : content) {
    if (!part.is_object() || !part.contains("type") ||
        !part["type"].is_string()) {
      throw std::runtime_error("Chat content parts must include a string type");
    }

    const std::string type = part["type"].get<std::string>();
    if (type == "text") {
      if (part.contains("text") && part["text"].is_string())
        text += part["text"].get<std::string>();
    } else {
      throw std::runtime_error("Unsupported non-text chat content part: " +
                               type);
    }
  }

  return text;
}

bool shouldAddGenerationPrompt(const OrderedJson &messages,
                               const nlohmann::json &request,
                               const ChatTemplate::Options &options) {
  using GenerationPromptMode = ChatTemplate::Options::GenerationPromptMode;

  if (options.generation_prompt == GenerationPromptMode::Always)
    return true;

  if (options.generation_prompt == GenerationPromptMode::Never)
    return false;

  if (request.is_object() && request.contains("add_generation_prompt") &&
      request["add_generation_prompt"].is_boolean())
    return request["add_generation_prompt"].get<bool>();

  if (messages.empty())
    throw std::runtime_error("chat_input.messages must not be empty");

  const auto &last = messages.back();
  if (!last.is_object() || !last.contains("role") || !last["role"].is_string())
    throw std::runtime_error("Each chat message must include a string role");

  return last["role"].get<std::string>() != "assistant";
}

std::string functionGemmaTemplate() {
  return R"CHAT_TEMPLATE({%- macro escaped(value) -%}<escape>{{ value }}<escape>{%- endmacro -%}
{%- macro format_properties(properties) -%}
{%- for key, val in properties.items() -%}
{{- "," if not loop.first else "" -}}{{ key }}:{
{%- set ns = namespace(first=true) -%}
{%- if val.description -%}
{{- "," if not ns.first else "" -}}description:{{ escaped(val.description) }}{%- set ns.first = false -%}
{%- endif -%}
{%- if val.type -%}
{{- "," if not ns.first else "" -}}type:{{ escaped(val.type | upper) }}{%- set ns.first = false -%}
{%- endif -%}
{%- if val.properties -%}
{{- "," if not ns.first else "" -}}properties:{ {{- format_properties(val.properties) -}} }{%- set ns.first = false -%}
{%- endif -%}
{%- if val.required -%}
{{- "," if not ns.first else "" -}}required:[
{%- for req in val.required -%}
{{- "," if not loop.first else "" -}}{{ escaped(req) }}
{%- endfor -%}
]{%- set ns.first = false -%}
{%- endif -%}
}
{%- endfor -%}
{%- endmacro -%}
{%- macro format_function_declaration(tool) -%}
{%- set func = tool.function -%}
declaration:{{ func.name }},description:{{ escaped(func.description) }},parameters:{
{%- set params = func.parameters -%}
{%- set ns = namespace(first=true) -%}
{%- if params.properties -%}
properties:{ {{- format_properties(params.properties) -}} }{%- set ns.first = false -%}
{%- endif -%}
{%- if params.required -%}
{{- "," if not ns.first else "" -}}required:[
{%- for req in params.required -%}
{{- "," if not loop.first else "" -}}{{ escaped(req) }}
{%- endfor -%}
]{%- set ns.first = false -%}
{%- endif -%}
{%- if params.type -%}
{{- "," if not ns.first else "" -}}type:{{ escaped(params.type | upper) }}
{%- endif -%}
}
{%- endmacro -%}
{{- bos_token -}}
{%- set ns = namespace(tools_inserted=false) -%}
{%- for message in messages -%}
{%- set role = "model" if message.role == "assistant" else message.role -%}
{%- if role != "tool" -%}
{{- "<start_of_turn>" + role + "\n" -}}
{%- endif -%}
{%- if role != "tool" and message.content is string -%}
{{- message.content -}}
{%- endif -%}
{%- if not ns.tools_inserted and tools and (role == "developer" or role == "system") -%}
{%- for tool in tools -%}
{{- "<start_function_declaration>" -}}{{ format_function_declaration(tool) }}{{- "<end_function_declaration>" -}}
{%- endfor -%}
{%- set ns.tools_inserted = true -%}
{%- endif -%}
{%- if message.tool_calls -%}
{%- for tool_call in message.tool_calls -%}
{%- set func = tool_call.function -%}
{{- "<start_function_call>call:" + func.name + "{" -}}
{%- if func.arguments is string -%}
{{- func.arguments -}}
{%- elif func.arguments -%}
{%- for key, val in func.arguments.items() -%}
{{- "," if not loop.first else "" -}}{{ key }}:{%- if val is string -%}{{ val }}{%- else -%}{{ val | tojson }}{%- endif -%}
{%- endfor -%}
{%- endif -%}
{{- "}<end_function_call>" -}}
{%- endfor -%}
{%- endif -%}
{%- if role != "tool" -%}
{{- "<end_of_turn>\n" -}}
{%- elif message.content -%}
{{- "<start_function_response>response:" + message.name + "{value:" -}}
{%- if message.content is string -%}{{ message.content }}{%- else -%}{{ message.content | tojson }}{%- endif -%}
{{- "}<end_function_response>" -}}
{%- endif -%}
{%- endfor -%}
{{- "<start_of_turn>model\n" -}})CHAT_TEMPLATE";
}

} // namespace

struct ChatTemplate::Impl {
  std::string model_path;
  std::string source_path;
  std::string template_source;
  nlohmann::json template_map = nlohmann::json::object();
  nlohmann::json tokenizer_config = nlohmann::json::object();
  nlohmann::json special_tokens = nlohmann::json::object();
  std::string bos_token;
  std::string eos_token;
  bool apply_polyfills = true;
  Options::DeveloperRolePolicy default_developer_role_policy =
    Options::DeveloperRolePolicy::MergeIntoSystem;
  mutable std::unordered_map<std::string,
                             std::unique_ptr<minja::chat_template>>
    renderers;

  const std::string &selectTemplate(const nlohmann::json &request,
                                    const Options &options,
                                    std::string &cache_key) const {
    if (!template_source.empty()) {
      cache_key = "__default__";
      return template_source;
    }

    if (!options.template_name.empty()) {
      if (!template_map.contains(options.template_name) ||
          !template_map[options.template_name].is_string()) {
        throw std::runtime_error("Requested chat template is not available: " +
                                 options.template_name);
      }
      cache_key = options.template_name;
      return template_map[options.template_name].get_ref<const std::string &>();
    }

    if (hasArrayItems(request, "tools") && template_map.contains("tool_use") &&
        template_map["tool_use"].is_string()) {
      cache_key = "tool_use";
      return template_map["tool_use"].get_ref<const std::string &>();
    }

    if (template_map.contains("default") &&
        template_map["default"].is_string()) {
      cache_key = "default";
      return template_map["default"].get_ref<const std::string &>();
    }

    for (auto it = template_map.begin(); it != template_map.end(); ++it) {
      if (it.value().is_string()) {
        cache_key = it.key();
        return it.value().get_ref<const std::string &>();
      }
    }

    throw std::runtime_error("No usable chat template was found");
  }

  const minja::chat_template &rendererFor(const std::string &key,
                                          const std::string &source) const {
    auto it = renderers.find(key);
    if (it != renderers.end())
      return *it->second;

    auto renderer =
      std::make_unique<minja::chat_template>(source, bos_token, eos_token);
    auto inserted = renderers.emplace(key, std::move(renderer));
    return *inserted.first->second;
  }

  OrderedJson normalizeMessages(const nlohmann::json &request,
                                const Options &options) const {
    const nlohmann::json *messages_json = nullptr;
    if (request.is_array()) {
      messages_json = &request;
    } else if (request.is_object() && request.contains("messages")) {
      messages_json = &request["messages"];
    }

    if (messages_json == nullptr || !messages_json->is_array())
      throw std::runtime_error("chat_input must contain a messages array");

    OrderedJson messages = OrderedJson::array();
    std::unordered_map<std::string, std::string> tool_call_names;
    Options::DeveloperRolePolicy developer_role_policy =
      options.developer_role_policy == Options::DeveloperRolePolicy::Auto
        ? default_developer_role_policy
        : options.developer_role_policy;

    for (const auto &message_json : *messages_json) {
      if (!message_json.is_object() || !message_json.contains("role") ||
          !message_json["role"].is_string()) {
        throw std::runtime_error(
          "Each chat message must include a string role");
      }

      OrderedJson message = toOrderedJson(message_json);
      std::string role = message_json["role"].get<std::string>();
      if (role == "developer" &&
          developer_role_policy ==
            Options::DeveloperRolePolicy::MergeIntoSystem) {
        message["role"] = "system";
      } else if (role != "system" && role != "user" && role != "assistant" &&
                 role != "tool" && role != "developer") {
        throw std::runtime_error("Unsupported chat message role: " + role);
      }

      if (message_json.contains("content") &&
          message_json["content"].is_array())
        message["content"] = textFromContentParts(message_json["content"]);

      if (message_json.contains("function_call") &&
          !message_json.contains("tool_calls")) {
        message["tool_calls"] = OrderedJson::array(
          {{{"id", "function_call"}, {"type", "function"},
            {"function", toOrderedJson(message_json["function_call"])}}});
      }

      if (message.contains("tool_calls") && message["tool_calls"].is_array()) {
        for (const auto &tool_call : message["tool_calls"]) {
          if (tool_call.contains("id") && tool_call["id"].is_string() &&
              tool_call.contains("function") &&
              tool_call["function"].contains("name") &&
              tool_call["function"]["name"].is_string()) {
            tool_call_names[tool_call["id"].get<std::string>()] =
              tool_call["function"]["name"].get<std::string>();
          }
        }
      }

      if (message["role"] == "tool" && !message.contains("name") &&
          message.contains("tool_call_id") &&
          message["tool_call_id"].is_string()) {
        const auto it =
          tool_call_names.find(message["tool_call_id"].get<std::string>());
        if (it != tool_call_names.end())
          message["name"] = it->second;
      }

      messages.push_back(std::move(message));
    }

    return messages;
  }

  OrderedJson buildExtraContext(const nlohmann::json &request) const {
    OrderedJson extra_context = OrderedJson::object();

    for (const auto &name : {"pad_token", "unk_token"}) {
      std::string token = findToken(tokenizer_config, special_tokens, name);
      if (!token.empty())
        extra_context[name] = token;
    }

    if (special_tokens.contains("additional_special_tokens")) {
      extra_context["additional_special_tokens"] =
        toOrderedJson(special_tokens["additional_special_tokens"]);
    } else if (tokenizer_config.contains("additional_special_tokens")) {
      extra_context["additional_special_tokens"] =
        toOrderedJson(tokenizer_config["additional_special_tokens"]);
    }

    if (request.is_object()) {
      for (auto it = request.begin(); it != request.end(); ++it) {
        const std::string &key = it.key();
        if (key == "messages" || key == "tools" ||
            key == "add_generation_prompt" ||
            key == "continue_final_message") {
          continue;
        }
        extra_context[key] = toOrderedJson(it.value());
      }
    }

    return extra_context;
  }
};

ChatTemplate::ChatTemplate(std::unique_ptr<Impl> impl) :
  impl_(std::move(impl)) {
}

ChatTemplate::ChatTemplate(ChatTemplate &&) noexcept = default;

ChatTemplate &ChatTemplate::operator=(ChatTemplate &&) noexcept = default;

ChatTemplate::~ChatTemplate() = default;

bool ChatTemplate::Exists(const std::string &model_path) {
  if (fileExists(model_path + "/chat_template.jinja"))
    return true;

  const std::string tokenizer_config_path =
    model_path + "/tokenizer_config.json";
  if (!fileExists(tokenizer_config_path))
    return false;

  try {
    nlohmann::json tokenizer_config = readJsonFile(tokenizer_config_path);
    return tokenizer_config.contains("chat_template") &&
           !tokenizer_config["chat_template"].is_null();
  } catch (...) {
    return false;
  }
}

ChatTemplate ChatTemplate::Load(const std::string &model_path) {
  auto impl = std::unique_ptr<Impl>(new Impl());
  impl->model_path = model_path;

  const std::string tokenizer_config_path =
    model_path + "/tokenizer_config.json";
  if (fileExists(tokenizer_config_path))
    impl->tokenizer_config = readJsonFile(tokenizer_config_path);

  const std::string special_tokens_path =
    model_path + "/special_tokens_map.json";
  if (fileExists(special_tokens_path))
    impl->special_tokens = readJsonFile(special_tokens_path);

  impl->bos_token =
    findToken(impl->tokenizer_config, impl->special_tokens, "bos_token");
  impl->eos_token =
    findToken(impl->tokenizer_config, impl->special_tokens, "eos_token");

  const std::string template_file_path = model_path + "/chat_template.jinja";
  if (fileExists(template_file_path)) {
    impl->source_path = template_file_path;
    impl->template_source = readTextFile(template_file_path);
    return ChatTemplate(std::move(impl));
  }

  if (!impl->tokenizer_config.contains("chat_template") ||
      impl->tokenizer_config["chat_template"].is_null()) {
    throw std::runtime_error("No chat template found under: " + model_path);
  }

  impl->source_path = tokenizer_config_path + ":chat_template";
  const auto &chat_template = impl->tokenizer_config["chat_template"];
  if (chat_template.is_string()) {
    impl->template_source = chat_template.get<std::string>();
  } else if (chat_template.is_object()) {
    impl->template_map = chat_template;
  } else {
    throw std::runtime_error("tokenizer_config.chat_template must be string or "
                             "object");
  }

  return ChatTemplate(std::move(impl));
}

ChatTemplate ChatTemplate::LoadBuiltin(Builtin builtin) {
  auto impl = std::unique_ptr<Impl>(new Impl());

  switch (builtin) {
  case Builtin::FunctionGemma:
    impl->source_path = "builtin:function_gemma";
    impl->template_source = functionGemmaTemplate();
    impl->bos_token = "<bos>";
    impl->eos_token = "<eos>";
    impl->apply_polyfills = false;
    impl->default_developer_role_policy =
      Options::DeveloperRolePolicy::Preserve;
    break;
  }

  return ChatTemplate(std::move(impl));
}

std::string ChatTemplate::apply(const nlohmann::json &request) const {
  return apply(request, Options());
}

std::string ChatTemplate::apply(const nlohmann::json &request,
                                const Options &options) const {
  if (!impl_)
    throw std::runtime_error("ChatTemplate is not initialized");

  if (options.continue_final_message ||
      (request.is_object() && request.value("continue_final_message", false))) {
    throw std::runtime_error(
      "continue_final_message is not supported by this runner yet");
  }

  OrderedJson messages = impl_->normalizeMessages(request, options);
  OrderedJson tools =
    request.is_object() && request.contains("tools")
      ? toOrderedJson(request["tools"])
      : OrderedJson::array();

  minja::chat_template_inputs inputs;
  inputs.messages = messages;
  inputs.tools = tools;
  inputs.add_generation_prompt =
    shouldAddGenerationPrompt(messages, request, options);
  inputs.extra_context = impl_->buildExtraContext(request);

  std::string cache_key;
  const std::string &source =
    impl_->selectTemplate(request, options, cache_key);
  minja::chat_template_options render_options;
  render_options.apply_polyfills = impl_->apply_polyfills;
  return impl_->rendererFor(cache_key, source).apply(inputs, render_options);
}

const std::string &ChatTemplate::sourcePath() const {
  if (!impl_)
    throw std::runtime_error("ChatTemplate is not initialized");

  return impl_->source_path;
}

} // namespace causallm
