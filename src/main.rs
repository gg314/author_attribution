use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::fmt;

mod nlp;

#[derive(Debug)]
struct CorpusData {
    name: String,
    total_sentences: i16,
    total_commas: i32,
    words_per_sentence: Vec<i16>,
    pronouns_per_sentence: Vec<i16>,
    conjunctions_per_sentence: Vec<i16>,
    word_frequencies: HashMap<String, i16>
}

#[derive(Debug)]
struct CorpusStats {
    name: String,
    total_sentences: i16,
    total_commas: i32,
    total_words: i32,
    hapax_legomena: i32,
    dis_legomena: i32,
    unique_words: i32,
    sentence_length_dist: [f64; 36],
    word_length_dist: [f64; 26],
    pronouns_per_sentence_dist: [f64; 20],
    conjunctions_per_sentence_dist: [f64; 20],
}

impl fmt::Display for CorpusStats {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut word_length_string = String::new();
        let mut sentence_length_string = String::new();
        let mut pronoun_dist_string = String::new();
        let mut conjunctions_dist_string = String::new();
        for l in self.word_length_dist.iter() {
            word_length_string = format!("{}  {:.2}", word_length_string, l);
        }
        for l in self.sentence_length_dist.iter() {
            sentence_length_string = format!("{}  {:.2}", sentence_length_string, l);
        }
        for l in self.pronouns_per_sentence_dist.iter() {
            pronoun_dist_string = format!("{}  {:.2}", pronoun_dist_string, l);
        }
        for l in self.conjunctions_per_sentence_dist.iter() {
            conjunctions_dist_string = format!("{}  {:.2}", conjunctions_dist_string, l);
        }
        
        write!(f, "=====================================================================================================================\n")?;
        write!(f, "|  Dataset: {}\n", self.name)?;
        write!(f, "=====================================================================================================================\n")?;
        write!(f, "|  unique words: {}   |   hapax legomena: {}   |   dis legomena: {}   |   commas: {} \n", self.unique_words, self.hapax_legomena, self.dis_legomena, self.total_commas)?;
        write!(f, "|  word lengths:              {} \n", word_length_string)?;
        write!(f, "|  sentence lengths:          {} \n", sentence_length_string)?;
        write!(f, "|  pronouns per sentence:     {} \n", pronoun_dist_string)?;
        write!(f, "|  conjunctions per sentence: {} \n", conjunctions_dist_string)?;
        write!(f, "\n")
    }
}


fn dot(v1: &Vec<i16>, v2: &Vec<i16>) -> i16 {
    return v1.iter().zip(v2.iter()).map(|(x, y)| x * y).sum();
}

fn norm(vin: &Vec<i16>) -> Vec<f64> {
    let mut sum = 0;
    for i in vin {
        sum += i*i;
    }

    let sum = (sum as f64).sqrt();
    let mut vout = Vec::new();
    for i in vin {
        vout.push( *i as f64 / sum )
    }

    return vout
}


fn statistics(corpus_data: CorpusData) -> CorpusStats {

    let sentence_lengths = corpus_data.words_per_sentence;
    let pronouns_per_sentence = corpus_data.pronouns_per_sentence;
    let conjunctions_per_sentence = corpus_data.conjunctions_per_sentence;
    let word_frequencies = &corpus_data.word_frequencies;

    let mut total_words: i32 = 0;
    let mut hapax_legomena = 0; // (fraction of) words used once
    let mut dis_legomena = 0; // (fraction of) words used twice
    let mut unique_words = 0;
    let mut sentence_length_dist: [i16; 36] = [0; 36];
    let mut word_length_dist: [i32; 26] = [0; 26];
    let mut pronouns_per_sentence_dist: [i16; 20] = [0; 20];
    let mut conjunctions_per_sentence_dist: [i16; 20] = [0; 20];

    
    for (word, count) in &*word_frequencies {
        let length: usize = word.chars().count();
        if length > 26 {
           // println!("Long word alert: {:?}", word);
            word_length_dist[25] += 1;
        } else if length > 0 {
            word_length_dist[length-1] += 1;
        } else {
            continue;
        }

        unique_words += 1;
        total_words += *count as i32;
        if *count == 1 {
            hapax_legomena += 1;
        } else if *count == 2 {
            dis_legomena += 1;
        }
    }
    // word_counts.clear();
    
    for l in sentence_lengths {
        let l = l as usize;
        if l > 3*36 {
            sentence_length_dist[35] += 1;
        } else if l > 0 {
            sentence_length_dist[(l-1)/3] += 1;
        } else {
            continue;
        }
    }
    
    for l in pronouns_per_sentence {
        let l = l as usize;
        if l > 19 {
            pronouns_per_sentence_dist[19] += 1;
        } else {
            pronouns_per_sentence_dist[l] += 1;
        }
    }
    
    for l in conjunctions_per_sentence {
        let l = l as usize;
        if l > 19 {
            conjunctions_per_sentence_dist[19] += 1;
        } else {
            conjunctions_per_sentence_dist[l] += 1;
        }
    }
    
    println!("total_words is {:?}", total_words);
    println!("hapax_legomena is {:?}", hapax_legomena);
    println!("dis_legomena is {:?}", dis_legomena);
    println!("unique_words is {:?}", unique_words);
    println!("sentence length dist is {:?}", sentence_length_dist);
    println!("word length dist is {:?}", word_length_dist);
    println!("pronoun dist is {:?}", pronouns_per_sentence_dist);
    println!("conjunction dist is {:?}", conjunctions_per_sentence_dist);

    
    let mut sentence_length_dist_f64: [f64; 36] = [0.0; 36];
    let mut word_length_dist_f64: [f64; 26] = [0.0; 26];
    let mut pronouns_per_sentence_dist_f64: [f64; 20] = [0.0; 20];
    let mut conjunctions_per_sentence_dist_f64: [f64; 20] = [0.0; 20];

    for (i, e) in sentence_length_dist.iter().enumerate() {
        sentence_length_dist_f64[i] = (*e as f64) / (corpus_data.total_sentences as f64);
    }
    
    for (i, e) in word_length_dist.iter().enumerate() {
        word_length_dist_f64[i] = (*e as f64) / (unique_words as f64);
    }
    
    for (i, e) in pronouns_per_sentence_dist.iter().enumerate() {
        pronouns_per_sentence_dist_f64[i] = (*e as f64) / (corpus_data.total_sentences as f64);
    }
    
    for (i, e) in conjunctions_per_sentence_dist.iter().enumerate() {
        conjunctions_per_sentence_dist_f64[i] = (*e as f64) / (corpus_data.total_sentences as f64);
    }

    CorpusStats {
        name: corpus_data.name,
        total_sentences: corpus_data.total_sentences,
        total_commas: corpus_data.total_commas,
        total_words: total_words,
        hapax_legomena: hapax_legomena,
        dis_legomena: dis_legomena,
        unique_words: unique_words,
        sentence_length_dist: sentence_length_dist_f64,
        word_length_dist: word_length_dist_f64,
        pronouns_per_sentence_dist: pronouns_per_sentence_dist_f64,
        conjunctions_per_sentence_dist: conjunctions_per_sentence_dist_f64,
    }
}


fn readit() -> CorpusData {
    // let filename = "data/test.txt";
    let filename = "data/corpus/unknown_pyrates_ii.txt";

    let mut total_sentences = 0;
    let mut count_words = 0;
    let mut count_pronouns = 0;
    let mut count_conjunctions = 0;
    let mut count_commas = 0;

    let file = File::open(filename).expect("no such file");
    let buf = BufReader::new(file);
    let text: Vec<String> = buf.lines().map(|l| l.expect("Could not parse line")).collect();
    let mut words_per_sentence: Vec<i16> = Vec::new();
    let mut pronouns_per_sentence: Vec<i16> = Vec::new();
    let mut conjunctions_per_sentence: Vec<i16> = Vec::new();
    let mut map: HashMap<String, i16> = HashMap::new();

    let mut test_sentence: String = String::new();

    for line in text {
        for word in line.split_whitespace() {
            let mut keyword = word.to_lowercase().to_string();
            keyword = keyword.replace(&['(', ')', '\"', ';', ':', '\'', '_'][..], "");
            count_words += 1;
            test_sentence.push_str(&keyword);
            test_sentence.push_str(" ");


            if keyword.contains(',') {
                count_commas += 1;
                keyword = keyword.replace(&[','][..], "");
            }

            keyword = nlp::replace_word(keyword);

            if keyword.contains('.') ||  keyword.contains('?') ||  keyword.contains('!') {
                keyword = keyword.replace(&['.', '?', '!'][..], "");
                total_sentences += 1;
                words_per_sentence.push(count_words);
                pronouns_per_sentence.push(count_pronouns);
                conjunctions_per_sentence.push(count_conjunctions);
                count_words = 0;
                count_pronouns = 0;
                count_conjunctions = 0;
                test_sentence = String::new();
            }

            if nlp::is_pronoun(word) {
                count_pronouns += 1;
            } 

            if nlp::is_conjunction(word) {
                count_conjunctions += 1;
            } 

            let count = map.entry(keyword).or_insert(0);
            *count += 1;
        }
    }
    
    CorpusData {
        name: String::from("pyrates"),
        total_sentences: total_sentences,
        total_commas: count_commas,
        words_per_sentence: words_per_sentence,
        pronouns_per_sentence: pronouns_per_sentence,
        conjunctions_per_sentence: conjunctions_per_sentence,
        word_frequencies: map
    }
}

fn main() {



    let v1 = vec![1, 2, 2, 4, 5];
    let v2 = vec![1, 1, 0, 1, 1];
    let v3 = dot(&v1, &v2);
    let v4 = norm(&v1);
    println!("dot prod is {:?}", v3);
    println!("the norm is {:?}", v4);

    let pyrates_data = readit();
    let pyrates_stats = statistics(pyrates_data);
    println!("{:}", pyrates_stats);


    // println!("{:?}", word_frequency);
}

