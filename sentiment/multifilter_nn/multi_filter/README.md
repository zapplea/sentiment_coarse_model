## what is new?
In previous program, the attribute function is max(a-x)e where e is word embedding of one word. This is equal to CNN with 1 pixel sized filter.
But in this program, we add more filter and their size is greater than one word.

## how to eliminate influence of #PAD# to convolution of multi-filter.

## Changes in use data generator
Nothing has changed compared with data generator to models in sep_nn, but need to copy the sep_nn's data generator to sentiment/util/multifilter/.

## Which data set does we use?
semeval2016 task5