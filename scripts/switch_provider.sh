#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <provider> [model]"
  echo "Providers: openai | groq | cerebras | google"
  echo "Examples:"
  echo "  $0 groq"
  echo "  $0 google gemini-2.5-flash-lite"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

provider="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
model_override="${2:-}"
env_file=".env"

if [[ ! -f "${env_file}" ]]; then
  echo "ERROR: ${env_file} not found in current directory."
  exit 1
fi

case "${provider}" in
  openai)
    model_key="OPENAI_MODEL"
    ;;
  groq)
    model_key="GROQ_MODEL"
    ;;
  cerebras)
    model_key="CEREBRAS_MODEL"
    ;;
  google)
    model_key="GOOGLE_MODEL"
    ;;
  *)
    echo "ERROR: Unsupported provider '${provider}'."
    usage
    exit 1
    ;;
esac

tmp_file="$(mktemp)"
awk -v p="${provider}" '
  BEGIN { done = 0 }
  /^AI_PROVIDER=/ {
    print "AI_PROVIDER=" p
    done = 1
    next
  }
  { print }
  END {
    if (done == 0) {
      print "AI_PROVIDER=" p
    }
  }
' "${env_file}" > "${tmp_file}"
mv "${tmp_file}" "${env_file}"

if [[ -n "${model_override}" ]]; then
  tmp_file="$(mktemp)"
  awk -v key="${model_key}" -v val="${model_override}" '
    BEGIN { done = 0 }
    $0 ~ ("^" key "=") {
      print key "=" val
      done = 1
      next
    }
    { print }
    END {
      if (done == 0) {
        print key "=" val
      }
    }
  ' "${env_file}" > "${tmp_file}"
  mv "${tmp_file}" "${env_file}"
fi

echo "Switched AI_PROVIDER to ${provider}."
if [[ -n "${model_override}" ]]; then
  echo "Updated ${model_key} to ${model_override}."
fi
