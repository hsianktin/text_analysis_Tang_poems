# Readme

This repository is a realization of text analysis based on the introduction to mini project in PKU CS101, for CCME freshmen. It's incomplete and welcome any further work based on my repository. I don't really have time to adjust code for readability. Sorry for the inconvenience.

## Reference
用文本挖掘剖析近5万首《全唐诗》 http://www.woshipm.com/data-analysis/970466.html


Things got stuck when trying to realize w2v network.
There are two obstacles. First, initialize a large matrix is memory-expensive. I used Google's Colab to help training. But the memory still exceeds the limit. Second, I used the standard network and 1000 poems to iterate for 3000 steps. The outcome is still unpromising.

## Known Issues
[ ] Network analysis is lost.

[ ] Word to vector mapping is far from ideal.