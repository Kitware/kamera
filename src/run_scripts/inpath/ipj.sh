#!/usr/bin/env bash

# Convert output of `ip` to something jq parseable

#SED commands
SC_columnToComma='s/ \+/,/g'           # foo     bar      spam > foo,bar,spam,
SC_dropLastComma='s/(.*),$/\1/'        # foo,bar,spam, > foo,bar,spam
SC_quoteFields='s/([^,]+)/\"\1\"/g'    # foo,bar,spam > "foo","bar","spam"
SC_encloseLine='s/(.*)/\[\1\]/g'       # "foo","bar","spam" > ["foo","bar","spam"]
SC_stripBadComma='s/\[(.*),]/\[\1\]/g' # ["foo","bar",] > ["foo","bar"]
JQ_mergeObjects='. | map( { (.name|tostring): . } ) | add'

# I have only half a clue how a quarter of this works
ip -4 -br -o a \
    | sed "$SC_columnToComma" \
    | sed -r "$SC_quoteFields" \
    | sed -r "$SC_encloseLine" \
    | sed -r "$SC_stripBadComma" \
    | jq '{name: .[0], status: .[1], ip: .[2]}' \
    | jq -s "$JQ_mergeObjects"
