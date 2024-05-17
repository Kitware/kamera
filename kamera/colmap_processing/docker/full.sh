# Location of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source $DIR/feature_extractor.sh $1
source $DIR/vocab_tree_matcher.sh $1
#source $DIR/exhaustive_matcher.sh $1
mkdir $DIR/sparse
mkdir $DIR/snapshots
source $DIR/mapper.sh $1
