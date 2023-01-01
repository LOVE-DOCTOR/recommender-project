mkdir -p ~/ .streamlit/

# shellcheck disable=SC2028
echo "\
[general]\n\
email=\"tunexo885@gmail.com\"\n\
" > ~/ .streamlit/credential.toml

# shellcheck disable=SC2028
echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/ .streamlit/config.toml