function [ stemmedBag ] = stemmer( bagOfWords )

% bagOfWords: 1 by N cell
% stemmedBag: 1 by N cell

stemmedBag = cell(1,length(bagOfWords));
for i=1:length(bagOfWords)
    stemmedBag(i) = {porterStemmer(char(bagOfWords(i)))};
end

end

