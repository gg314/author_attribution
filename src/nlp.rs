pub fn is_pronoun(w: &str) -> bool {
    match w {
        "i" => true,
        "you" => true,
        "he" => true,
        "she" => true,
        "we" => true,
        "they" => true,
        "it" => true,
        "whoever" => true,
        _ => false
    }
}

pub fn is_conjunction(w: &str) -> bool {
    match w {
        "and" => true,
        "but" => true,
        "or" => true,
        "nor" => true,
        "for" => true,
        "yet" => true,
        "so" => true,
        "after" => true,
        "although" => true,
        "as" => true,
        "as if" => true,
        "as long as" => true,
        "as much as" => true,
        "as soon as" => true,
        "as though" => true,
        "because" => true,
        "before" => true,
        "even " => true,
        "even if" => true,
        "even though" => true,
        "if" => true,
        "if only" => true,
        "if when" => true,
        "if then" => true,
        "inasmuch" => true,
        "in order that" => true,
        "just as" => true,
        "lest" => true,
        "now" => true,
        "now since" => true,
        "now that" => true,
        "now when" => true,
        "once" => true,
        "provided" => true,
        "provided that" => true,
        "rather than" => true,
        "since" => true,
        "so that" => true,
        "supposing" => true,
        "than" => true,
        "that" => true,
        "though" => true,
        "til" => true,
        "unless" => true,
        "until" => true,
        "when" => true,
        "whenever" => true,
        "where" => true,
        "whereas" => true,
        "where if" => true,
        "wherever" => true,
        "whether" => true,
        "which" => true,
        "while" => true,
        "who" => true,
        "whoever" => true,
        "why" => true,
        _ => false
    }
}

pub fn replace_word(w: String) -> String {
    let resp =
        match &w[..] {
            "&c." => "etc",
            "wm." => "william",
            "tho." => "thomas",
            "geo." => "george",
            "st." => "saint",
            "vol." => "volume",
            "capt." => "captain",
            "mr." => "mister",
            "mrs." => "missus",
            "col." => "colonel",
            "gen." => "general",
            "mjr." => "major",
            "lat." => "latitude",
            "lon." => "longitude",
            "a." => "",
            "b." => "",
            "c." => "",
            "d." => "",
            "e." => "",
            "f." => "",
            "g." => "",
            "h." => "",
            "i." => "",
            "j." => "",
            "k." => "",
            "l." => "",
            "m." => "",
            "n." => "",
            "o." => "",
            "p." => "",
            "q." => "",
            "r." => "",
            "s." => "",
            "t." => "",
            "u." => "",
            "v." => "",
            "w." => "",
            "x." => "",
            "y." => "",
            "z." => "",
            other => other
        };
    return resp.to_owned()
}