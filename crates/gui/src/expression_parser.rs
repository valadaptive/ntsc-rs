//! This is a simple parser for math expressions so that users can enter them into the GUI's slider numeric inputs.

use logos::{Lexer, Logos};
use std::mem;

fn parse_num(lex: &mut Lexer<Token>) -> Option<f64> {
    lex.slice().parse::<f64>().ok()
}

#[derive(Clone, Copy, Logos, Debug, PartialEq)]
#[logos(skip r"[ \t\n\f]+")]
enum Token {
    #[regex(r"([0-9]+(\.[0-9]*)?|(\.[0-9]+))(e[+-][0-9]+)?", parse_num)]
    Number(f64),

    #[token("+")]
    Plus,

    #[token("-")]
    Minus,

    #[token("*")]
    Multiply,

    #[token("/")]
    Divide,

    #[token("**")]
    Power,

    #[token("%")]
    Percent,

    #[token("(")]
    LParen,

    #[token(")")]
    RParen,
}

#[derive(Debug)]
pub struct ParseError {
    pub message: String,
}

impl ParseError {
    fn new(message: &str) -> Self {
        ParseError {
            message: String::from(message),
        }
    }
}

struct LexerWrapper<'a> {
    lexer: Lexer<'a, Token>,
    cur: Option<Token>,
    next: Option<Token>,
}

impl<'a> LexerWrapper<'a> {
    fn new_from_str(string: &'a str) -> Result<LexerWrapper, ParseError> {
        let mut lexer = Token::lexer(string);
        let cur = None;
        let next = lexer.next();
        if let Some(token) = &next {
            if token.is_err() {
                return Err(ParseError::new("aaa"));
            }
        };

        Ok(LexerWrapper {
            lexer,
            cur,
            next: next.map(|token| token.unwrap()),
        })
    }

    fn advance(&mut self) -> Result<Option<&Token>, ParseError> {
        let next = self
            .lexer
            .next()
            .transpose()
            .map_err(|_| ParseError::new("No next token"))?;
        let old_next = mem::replace(&mut self.next, next);
        self.cur = old_next;
        Ok(self.next.as_ref())
    }
}

fn prefix_binding_power(op: &Token) -> usize {
    match op {
        Token::Plus | Token::Minus => 7,
        _ => panic!("not a prefix operator: {op:?}"),
    }
}

fn postfix_binding_power(op: &Token) -> Option<usize> {
    match op {
        Token::Percent => Some(9),
        _ => None,
    }
}

fn infix_binding_power(op: &Token) -> Option<(usize, usize)> {
    match op {
        Token::Plus | Token::Minus => Some((1, 2)),
        Token::Multiply | Token::Divide => Some((3, 4)),
        Token::Power => Some((6, 5)),
        _ => None,
    }
}

fn eval_expr(lexer: &mut LexerWrapper, min_binding_power: usize) -> Result<f64, ParseError> {
    let mut lhs = match lexer.cur {
        Some(Token::LParen) => {
            lexer.advance()?;
            let res = eval_expr(lexer, 0)?;
            if lexer.cur != Some(Token::RParen) {
                return Err(ParseError::new("Expected closing parenthesis"));
            }
            Ok(res)
        }
        Some(Token::Number(value)) => Ok(value),
        Some(Token::Plus) => {
            let inner_value = eval_expr(lexer, prefix_binding_power(&Token::Plus))?;
            // unary plus does nothing
            Ok(inner_value)
        }
        Some(Token::Minus) => {
            let inner_value = eval_expr(lexer, prefix_binding_power(&Token::Minus))?;
            // unary negation
            Ok(-inner_value)
        }
        _ => Err(ParseError::new("Invalid left-hand side")),
    }?;

    lexer.advance()?;

    loop {
        let op = match lexer.cur {
            Some(token) => match token {
                Token::Plus
                | Token::Minus
                | Token::Multiply
                | Token::Divide
                | Token::Power
                | Token::Percent => Ok(token),
                Token::RParen => break Ok(lhs),
                _ => Err(ParseError::new("Expected operator")),
            },
            None => break Ok(lhs),
        }?;

        if let Some(left_bp) = postfix_binding_power(&op) {
            if left_bp < min_binding_power {
                break Ok(lhs);
            }
            lexer.advance()?;
            lhs = match op {
                Token::Percent => lhs * 0.01,
                _ => panic!("unhandled op: {op:?}"),
            };
            continue;
        }

        if let Some((left_bp, right_bp)) = infix_binding_power(&op) {
            if left_bp < min_binding_power {
                break Ok(lhs);
            }

            lexer.advance()?;
            let rhs = eval_expr(lexer, right_bp)?;

            lhs = match op {
                Token::Plus => lhs + rhs,
                Token::Minus => lhs - rhs,
                Token::Multiply => lhs * rhs,
                Token::Divide => lhs / rhs,
                Token::Power => lhs.powf(rhs),
                _ => panic!("unhandled op: {op:?}"),
            };
            continue;
        }

        break Ok(lhs);
    }
}

pub fn eval_expression_string(string: &str) -> Result<f64, ParseError> {
    let mut lexer = LexerWrapper::new_from_str(string)?;
    lexer.advance()?;

    eval_expr(&mut lexer, 0)
}
