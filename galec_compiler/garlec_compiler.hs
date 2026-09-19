module Main where

import Data.Char (isSpace, isDigit, isAlpha, isAlphaNum)
import Data.List (intercalate, nub)
import Control.Applicative (Alternative(empty, (<|>)))


sourceFile :: FilePath
sourceFile = "simple_galec.txt"


--------------------------------------------------------------------------------
-- Parser core
--------------------------------------------------------------------------------

data Reply a
    = Ok a String Int
    | Err [String] Int


newtype Parser a =
    Parser { runParser :: String -> Int -> Reply a }


instance Functor Parser where
    fmap :: (a -> b) -> Parser a -> Parser b
    fmap f p = Parser $ \s pos ->
        case runParser p s pos of
            Ok a s' pos'  -> Ok (f a) s' pos'
            Err msgs pos' -> Err msgs pos'


instance Applicative Parser where
    pure :: a -> Parser a
    pure a = Parser $ \s pos -> Ok a s pos
    (<*>) :: Parser (a -> b) -> Parser a -> Parser b
    pf <*> pa = Parser $ \s pos ->
        case runParser pf s pos of
            Err msgs pos' -> Err msgs pos'
            Ok f s' pos' ->
                case runParser pa s' pos' of
                    Err msgs pos'' -> Err msgs pos''
                    Ok a s'' pos'' -> Ok (f a) s'' pos''


instance Monad Parser where
    (>>=) :: Parser a -> (a -> Parser b) -> Parser b
    p >>= f = Parser $ \s pos ->
        case runParser p s pos of
            Err msgs pos' -> Err msgs pos'
            Ok a s' pos'  -> runParser (f a) s' pos'


instance Alternative Parser where
    empty = Parser $ \_ pos -> Err [] pos

    p <|> q = Parser $ \s pos ->
        case runParser p s pos of
            Ok a s' pos' -> Ok a s' pos'
            Err msgs1 pos1 ->
                case runParser q s pos of
                    Ok a s' pos' -> Ok a s' pos'
                    Err msgs2 pos2 ->
                        case compare pos1 pos2 of
                            GT -> Err msgs1 pos1
                            LT -> Err msgs2 pos2
                            EQ -> Err (nub (msgs1 ++ msgs2)) pos1


label :: String -> Parser a -> Parser a
label name p = Parser $ \s pos ->
    case runParser p s pos of
        Ok a s' pos'  -> Ok a s' pos'
        Err [] pos'   -> Err [name] pos'
        Err msgs pos' -> Err msgs pos'


--------------------------------------------------------------------------------
-- Primitive parsers
--------------------------------------------------------------------------------

satisfy :: String -> (Char -> Bool) -> Parser Char
satisfy desc predicate = Parser $ \s pos ->
    case s of
        [] -> Err [desc] pos
        (c:cs) ->
            if predicate c
                then Ok c cs (pos + 1)
                else Err [desc] pos


sat :: (Char -> Bool) -> Parser Char
sat = satisfy "character"


char :: Char -> Parser Char
char c = satisfy ("'" ++ [c] ++ "'") (== c)


string :: String -> Parser String
string s = label (show s) (go s)
  where
    go []     = return []
    go (c:cs) = do
        _ <- char c
        _ <- go cs
        return (c:cs)


eof :: Parser ()
eof = Parser $ \s pos ->
    case s of
        [] -> Ok () [] pos
        _  -> Err ["end of input"] pos


notFollowedBy :: Parser a -> Parser ()
notFollowedBy p = Parser $ \s pos ->
    case runParser p s pos of
        Ok _ _ _ -> Err ["unexpected input"] pos
        Err _ _  -> Ok () s pos


--------------------------------------------------------------------------------
-- Repetition
--------------------------------------------------------------------------------

many :: Parser a -> Parser [a]
many p = many1 p <|> return []


many1 :: Parser a -> Parser [a]
many1 p = do
    x  <- p
    xs <- many p
    return (x : xs)


optional :: Parser a -> Parser (Maybe a)
optional p = (Just <$> p) <|> return Nothing


--------------------------------------------------------------------------------
-- Lexer
--------------------------------------------------------------------------------

spaces :: Parser ()
spaces = do
    many spaceOrComment
    return ()


spaceOrComment :: Parser Char
spaceOrComment = satisfy "whitespace" isSpace <|> comment


comment :: Parser Char
comment = do
    _ <- string "//"
    _ <- many (satisfy "non-newline" (/= '\n'))
    _ <- optional (char '\n')
    return ' '


token :: Parser a -> Parser a
token p = do
    x <- p
    spaces
    return x


symbol :: String -> Parser String
symbol s = token (string s)


keyword :: String -> Parser String
keyword kw = do
    x <- label ("keyword " ++ show kw) (string kw)
    notFollowedBy (satisfy "identifier character"
                     (\c -> isAlphaNum c || c == '_'))
    spaces
    return x


reservedWords :: [String]
reservedWords =
    [ "block", "method", "algorithm", "end", "public"
    , "input", "output"
    , "Real", "Integer", "Boolean", "String"
    , "if", "then", "else"
    , "for", "in", "to", "do", "step"
    , "and", "or", "not", "true", "false"
    ]


identifier :: Parser String
identifier = token $ label "identifier" $
    Parser $ \s pos ->
        case runParser identRaw s pos of
            Ok name s' pos'
                | name `elem` reservedWords -> Err ["identifier"] pos
                | otherwise                -> Ok name s' pos'
            err -> err
  where
    identRaw = do
        first <- satisfy "letter" isAlpha
        rest  <- many (satisfy "alphanumeric or underscore"
                         (\c -> isAlphaNum c || c == '_'))
        return (first : rest)


--------------------------------------------------------------------------------
-- AST
--------------------------------------------------------------------------------

data Type      = Real | Integer | Boolean | StringT deriving Show
data Direction = Input | Output                     deriving Show
data Decl      = Decl Direction Type String         deriving Show

data Expr
    = Var String
    | IntLit Integer
    | RealLit Double
    | BoolLit Bool
    | Add Expr Expr
    | Sub Expr Expr
    | Mul Expr Expr
    | Div Expr Expr
    | Neg Expr
    | Not Expr
    | And Expr Expr
    | Or  Expr Expr
    | Eq  Expr Expr
    | Ne  Expr Expr
    | Lt  Expr Expr
    | Le  Expr Expr
    | Gt  Expr Expr
    | Ge  Expr Expr
    | Call String [Expr]
    deriving Show

data Stmt
    = Assign String Expr
    | If  Expr [Stmt] [Stmt]
    | For String Expr Expr (Maybe Expr) [Stmt]
    deriving Show

data Method = Method String [Stmt]         deriving Show
data Block  = Block String [Decl] [Method] deriving Show


--------------------------------------------------------------------------------
-- Grammar: declarations
--------------------------------------------------------------------------------

dataType :: Parser Type
dataType =
        Real    <$ keyword "Real"
    <|> Integer <$ keyword "Integer"
    <|> Boolean <$ keyword "Boolean"
    <|> StringT <$ keyword "String"


direction :: Parser Direction
direction =
        Input  <$ keyword "input"
    <|> Output <$ keyword "output"


decl :: Parser Decl
decl = do
    d <- direction
    t <- dataType
    n <- identifier
    _ <- symbol ";"
    return (Decl d t n)


qualifiedName :: Parser String
qualifiedName = do
    first <- identifier
    rest  <- optional $ do
        _ <- char '.'
        identifier
    case rest of
        Nothing -> return first
        Just x  -> return (first ++ "." ++ x)


--------------------------------------------------------------------------------
-- Grammar: expressions
--------------------------------------------------------------------------------

expression :: Parser Expr
expression = orExpr


orExpr :: Parser Expr
orExpr = chainl1 andExpr (Or <$ keyword "or")


andExpr :: Parser Expr
andExpr = chainl1 notExpr (And <$ keyword "and")


notExpr :: Parser Expr
notExpr =
        Not <$> (keyword "not" *> notExpr)
    <|> comparison


comparison :: Parser Expr
comparison = do
    left <- additive
    mop  <- optional compareOp
    case mop of
        Nothing -> return left
        Just op -> do
            right <- additive
            return (op left right)


compareOp :: Parser (Expr -> Expr -> Expr)
compareOp =
        Eq <$ symbol "="
    <|> Ne <$ symbol "<>"
    <|> Le <$ symbol "<="
    <|> Ge <$ symbol ">="
    <|> Lt <$ symbol "<"
    <|> Gt <$ symbol ">"


additive :: Parser Expr
additive = chainl1 term addOp


addOp :: Parser (Expr -> Expr -> Expr)
addOp = Add <$ symbol "+" <|> Sub <$ symbol "-"


term :: Parser Expr
term = chainl1 unary mulOp


mulOp :: Parser (Expr -> Expr -> Expr)
mulOp = Mul <$ symbol "*" <|> Div <$ symbol "/"


unary :: Parser Expr
unary =
        Neg <$> (symbol "-" *> unary)
    <|> atom


atom :: Parser Expr
atom =
        boolLit
    <|> number
    <|> parentheses
    <|> callOrAtom


callOrAtom :: Parser Expr
callOrAtom = do
    name  <- qualifiedName
    margs <- optional (symbol "(" *> argList <* symbol ")")
    case margs of
        Nothing   -> return (Var name)
        Just args -> return (Call name args)


argList :: Parser [Expr]
argList =
        (do first <- expression
            rest  <- many (symbol "," *> expression)
            return (first : rest))
    <|> return []


boolLit :: Parser Expr
boolLit =
        BoolLit True  <$ keyword "true"
    <|> BoolLit False <$ keyword "false"


number :: Parser Expr
number = token $ do
    whole <- many1 (satisfy "digit" isDigit)
    fraction <- optional $ do
        _ <- char '.'
        many1 (satisfy "digit" isDigit)
    case fraction of
        Nothing   -> return (IntLit (read whole))
        Just frac -> return (RealLit (read (whole ++ "." ++ frac)))


parentheses :: Parser Expr
parentheses = do
    _ <- symbol "("
    x <- expression
    _ <- symbol ")"
    return x


chainl1 :: Parser a -> Parser (a -> a -> a) -> Parser a
chainl1 p op = do
    first <- p
    rest first
  where
    rest x = (do f <- op
                 y <- p
                 rest (f x y))
             <|> return x


--------------------------------------------------------------------------------
-- Grammar: statements
--------------------------------------------------------------------------------

statement :: Parser Stmt
statement = ifStmt <|> forStmt <|> assignStmt


assignStmt :: Parser Stmt
assignStmt = do
    name  <- qualifiedName
    _     <- symbol ":="
    value <- expression
    _     <- symbol ";"
    return (Assign name value)


ifStmt :: Parser Stmt
ifStmt = do
    _         <- keyword "if"
    cond      <- expression
    _         <- keyword "then"
    thenStmts <- many statement
    elseStmts <- optional (keyword "else" *> many statement)
    _         <- keyword "end"
    _         <- keyword "if"
    _         <- symbol ";"
    return (If cond thenStmts (maybe [] id elseStmts))


forStmt :: Parser Stmt
forStmt = do
    _     <- keyword "for"
    var   <- identifier
    _     <- keyword "in"
    lo    <- expression
    _     <- keyword "to"
    hi    <- expression
    mstep <- optional (keyword "step" *> expression)
    _     <- keyword "do"
    body  <- many statement
    _     <- keyword "end"
    _     <- keyword "for"
    _     <- symbol ";"
    return (For var lo hi mstep body)


--------------------------------------------------------------------------------
-- Grammar: methods and blocks
--------------------------------------------------------------------------------

method :: Parser Method
method = do
    _          <- keyword "method"
    name       <- identifier
    _          <- keyword "algorithm"
    statements <- many statement
    _          <- keyword "end"
    endName    <- identifier
    _          <- symbol ";"
    if name == endName
        then return (Method name statements)
        else empty


block :: Parser Block
block = do
    _            <- keyword "block"
    name         <- identifier
    declarations <- many decl
    _            <- optional (keyword "public")
    methods      <- many method
    _            <- keyword "end"
    endName      <- identifier
    _            <- symbol ";"
    if name == endName
        then return (Block name declarations methods)
        else empty


galecFile :: Parser Block
galecFile = do
    spaces
    b <- block
    spaces
    eof
    return b


--------------------------------------------------------------------------------
-- Error formatting
--------------------------------------------------------------------------------

positionToLineCol :: String -> Int -> (Int, Int)
positionToLineCol src pos = go 1 1 0 src
  where
    go line col n (c:cs)
        | n >= pos  = (line, col)
        | c == '\n' = go (line + 1) 1 (n + 1) cs
        | otherwise = go line (col + 1) (n + 1) cs
    go line col _ [] = (line, col)


nthLine :: String -> Int -> String
nthLine src n =
    let lns = lines src
    in if n >= 1 && n <= length lns
          then lns !! (n - 1)
          else ""


formatError :: String -> Int -> [String] -> String
formatError src pos msgs =
    let (line, col) = positionToLineCol src pos
        content     = nthLine src line
        caret       = replicate (col - 1) ' ' ++ "^"
        expected
            | null msgs = "parse error"
            | otherwise = "expected " ++ intercalate " or " (nub msgs)
    in unlines
        [ "Error at line " ++ show line ++ ", column " ++ show col
              ++ " (offset " ++ show pos ++ ")"
        , expected
        , ""
        , content
        , caret
        ]


--------------------------------------------------------------------------------
-- AST tree printer
--------------------------------------------------------------------------------

indent :: Int -> String -> String
indent n s = replicate (n * 2) ' ' ++ s

prettyDir :: Direction -> String
prettyDir Input  = "input"
prettyDir Output = "output"

prettyType :: Type -> String
prettyType Real    = "Real"
prettyType Integer = "Integer"
prettyType Boolean = "Boolean"
prettyType StringT = "String"

printAST :: Block -> IO ()
printAST block =
    putStrLn (treeBlock block)


treeBlock :: Block -> String
treeBlock (Block name decls methods) =
    unlines $
        ["Block: " ++ name]
        ++ concatMap treeDecl decls
        ++ concatMap treeMethod methods


treeDecl :: Decl -> [String]
treeDecl (Decl dir ty name) =
    [ "  Declaration: "
      ++ prettyDir dir
      ++ " "
      ++ prettyType ty
      ++ " "
      ++ name
    ]


treeMethod :: Method -> [String]
treeMethod (Method name stmts) =
    [ "  Method: " ++ name ]
    ++ concatMap (treeStmt 2) stmts


treeStmt :: Int -> Stmt -> [String]

treeStmt d (Assign name expr) =
    [ indent d "Assign"
    , indent (d + 1) ("Variable: " ++ name)
    ]
    ++ treeExpr (d + 1) expr


treeStmt d (If cond thenStmts elseStmts) =
    [ indent d "If"
    , indent (d + 1) "condition"
    ]
    ++ treeExpr (d + 2) cond
    ++ [ indent (d + 1) "then" ]
    ++ concatMap (treeStmt (d + 2)) thenStmts
    ++ [ indent (d + 1) "else" ]
    ++ concatMap (treeStmt (d + 2)) elseStmts


treeStmt d (For var lo hi mstep body) =
    [ indent d ("For: " ++ var)
    , indent (d + 1) "from"
    ]
    ++ treeExpr (d + 2) lo
    ++ [ indent (d + 1) "to" ]
    ++ treeExpr (d + 2) hi
    ++ stepTree (d + 1) mstep
    ++ [ indent (d + 1) "body" ]
    ++ concatMap (treeStmt (d + 2)) body


stepTree :: Int -> Maybe Expr -> [String]
stepTree _ Nothing = []

stepTree d (Just expr) =
    [ indent d "step" ]
    ++ treeExpr (d + 1) expr


treeExpr :: Int -> Expr -> [String]

treeExpr d (Var name) =
    [ indent d ("Variable: " ++ name) ]

treeExpr d (IntLit n) =
    [ indent d ("Int: " ++ show n) ]

treeExpr d (RealLit x) =
    [ indent d ("Real: " ++ show x) ]

treeExpr d (BoolLit b) =
    [ indent d ("Boolean: " ++ show b) ]

treeExpr d (Neg e) =
    [ indent d "Neg" ]
    ++ treeExpr (d + 1) e

treeExpr d (Not e) =
    [ indent d "Not" ]
    ++ treeExpr (d + 1) e

treeExpr d (Add a b) =
    binaryTree d "Add" a b

treeExpr d (Sub a b) =
    binaryTree d "Sub" a b

treeExpr d (Mul a b) =
    binaryTree d "Mul" a b

treeExpr d (Div a b) =
    binaryTree d "Div" a b

treeExpr d (And a b) =
    binaryTree d "And" a b

treeExpr d (Or a b) =
    binaryTree d "Or" a b

treeExpr d (Eq a b) =
    binaryTree d "Eq" a b

treeExpr d (Ne a b) =
    binaryTree d "Ne" a b

treeExpr d (Lt a b) =
    binaryTree d "Lt" a b

treeExpr d (Le a b) =
    binaryTree d "Le" a b

treeExpr d (Gt a b) =
    binaryTree d "Gt" a b

treeExpr d (Ge a b) =
    binaryTree d "Ge" a b

treeExpr d (Call name args) =
    [ indent d ("Call: " ++ name) ]
    ++ concatMap (treeExpr (d + 1)) args


binaryTree :: Int -> String -> Expr -> Expr -> [String]
binaryTree d name a b =
    [ indent d name ]
    ++ treeExpr (d + 1) a
    ++ treeExpr (d + 1) b


--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

main :: IO ()
main = do
    source <- readFile sourceFile
    case runParser galecFile source 0 of
        Ok ast _ _   -> printAST ast
        Err msgs pos -> putStrLn (formatError source pos msgs)