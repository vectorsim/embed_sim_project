

--------------------------------------------------------------------------------
-- GALEC Compiler
-- Improved separation of concerns + monadic code generation
--------------------------------------------------------------------------------

module Main where

import Data.Char (isSpace, isDigit, isAlpha, isAlphaNum)
import Data.List (intercalate, nub)
import Control.Applicative (Alternative(empty, (<|>)), many, some, optional)
import Control.Monad (when, void)
import Control.Monad.Reader (Reader, runReader, ask, local, asks)
import System.Environment (getArgs)


defaultSource :: FilePath
defaultSource = "pmsm.galec"


--------------------------------------------------------------------------------
-- 1. PARSER CORE  (pure combinators – no domain knowledge)
--------------------------------------------------------------------------------

data Reply a
    = Ok a String Int
    | Err [String] Int
    deriving Show

newtype Parser a =
    Parser { runParser :: String -> Int -> Reply a }

instance Functor Parser where
    fmap f p = Parser $ \s pos ->
        case runParser p s pos of
            Ok a s' pos'  -> Ok (f a) s' pos'
            Err msgs pos' -> Err msgs pos'

instance Applicative Parser where
    pure a = Parser $ \s pos -> Ok a s pos
    pf <*> pa = Parser $ \s pos ->
        case runParser pf s pos of
            Err msgs pos' -> Err msgs pos'
            Ok f s' pos' ->
                case runParser pa s' pos' of
                    Err msgs pos'' -> Err msgs pos''
                    Ok a s'' pos'' -> Ok (f a) s'' pos''

instance Monad Parser where
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

-- Label a parser so error messages become more informative
label :: String -> Parser a -> Parser a
label name p = Parser $ \s pos ->
    case runParser p s pos of
        Ok a s' pos'  -> Ok a s' pos'
        Err [] pos'   -> Err [name] pos'
        Err msgs pos' -> Err msgs pos'


--------------------------------------------------------------------------------
-- 2. PRIMITIVE PARSERS
--------------------------------------------------------------------------------

satisfy :: String -> (Char -> Bool) -> Parser Char
satisfy desc predicate = Parser $ \s pos ->
    case s of
        []     -> Err [desc] pos
        (c:cs) -> if predicate c
                    then Ok c cs (pos + 1)
                    else Err [desc] pos

sat :: (Char -> Bool) -> Parser Char
sat = satisfy "character"

char :: Char -> Parser Char
char c = satisfy ("'" ++ [c] ++ "'") (== c)

string :: String -> Parser String
string s = label (show s) (go s)
  where
    go []     = pure []
    go (c:cs) = (:) <$> char c <*> go cs

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
-- 3. LEXER
--------------------------------------------------------------------------------

spaces :: Parser ()
spaces = void (many spaceOrComment)

spaceOrComment :: Parser Char
spaceOrComment = satisfy "whitespace" isSpace <|> comment

comment :: Parser Char
comment = do
    void (string "//")
    void (many (satisfy "non-newline" (/= '\n')))
    void (optional (char '\n'))
    pure ' '

token :: Parser a -> Parser a
token p = p <* spaces

symbol :: String -> Parser String
symbol s = token (string s)

keyword :: String -> Parser String
keyword kw = do
    void $ label ("keyword " ++ show kw) (string kw)
    notFollowedBy (satisfy "identifier character"
                     (\c -> isAlphaNum c || c == '_'))
    spaces
    pure kw

reservedWords :: [String]
reservedWords =
    [ "block", "method", "algorithm", "end", "public"
    , "input", "output", "parameter", "state", "function", "external"
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
                | otherwise                 -> Ok name s' pos'
            err -> err
  where
    identRaw = do
        first <- satisfy "letter" isAlpha
        rest  <- many (satisfy "alphanumeric or underscore"
                         (\c -> isAlphaNum c || c == '_'))
        pure (first : rest)

qualifiedName :: Parser String
qualifiedName = do
    first <- identifier
    rest  <- optional (char '.' *> identifier)
    pure $ maybe first (\x -> first ++ "." ++ x) rest


--------------------------------------------------------------------------------
-- 4. ABSTRACT SYNTAX TREE  (pure data – no behaviour)
--------------------------------------------------------------------------------

data Type      = Real | Integer | Boolean | StringT deriving (Show, Eq)
data Direction = Input | Output                     deriving (Show, Eq)

data Attr = Attr String Expr deriving Show

data Decl
    = Decl Direction Type String [Attr]
    | ParamDecl Type String
    | StateDecl Type String
    | InternalDecl Type String
    deriving Show

data Expr
    = Var String
    | IntLit Integer
    | RealLit Double
    | BoolLit Bool
    | Add Expr Expr | Sub Expr Expr
    | Mul Expr Expr | Div Expr Expr
    | Neg Expr | Not Expr
    | And Expr Expr | Or  Expr Expr
    | Eq  Expr Expr | Ne  Expr Expr
    | Lt  Expr Expr | Le  Expr Expr
    | Gt  Expr Expr | Ge  Expr Expr
    | Call String [Expr]
    deriving Show

data Stmt
    = Assign String Expr
    | MultiAssign [String] Expr
    | If  Expr [Stmt] [Stmt]
    | For String Expr Expr (Maybe Expr) [Stmt]
    deriving Show

data Method = Method String [Stmt]                  deriving Show
data Func   = Func String [Decl] String             deriving Show  -- name, decls, external name
data Block  = Block String [Decl] [Func] [Method]   deriving Show


--------------------------------------------------------------------------------
-- 5. GRAMMAR / PARSER  (builds the AST)
--------------------------------------------------------------------------------

-- Types & directions ----------------------------------------------------------

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

-- Attributes ------------------------------------------------------------------

attr :: Parser Attr
attr = Attr <$> identifier <*> (symbol "=" *> expression)

attrList :: Parser [Attr]
attrList = symbol "(" *> ((:) <$> attr <*> many (symbol "," *> attr)) <* symbol ")"

-- Declarations ----------------------------------------------------------------

ioDecl :: Parser Decl
ioDecl = do
    d     <- direction
    t     <- dataType
    n     <- identifier
    attrs <- optional attrList
    void (symbol ";")
    pure (Decl d t n (maybe [] id attrs))

paramDecl :: Parser Decl
paramDecl = ParamDecl
    <$  keyword "parameter"
    <*> dataType
    <*> identifier
    <*  symbol ";"

stateDecl :: Parser Decl
stateDecl = StateDecl
    <$  keyword "state"
    <*> dataType
    <*> identifier
    <*  symbol ";"

internalDecl :: Parser Decl
internalDecl = InternalDecl
    <$> dataType
    <*> identifier
    <*  symbol ";"

decl :: Parser Decl
decl = ioDecl <|> paramDecl <|> stateDecl <|> internalDecl

-- Expressions (precedence climbing) -------------------------------------------

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
        Nothing -> pure left
        Just op -> op left <$> additive

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
    pure $ maybe (Var name) (Call name) margs

argList :: Parser [Expr]
argList =
        ((:) <$> expression <*> many (symbol "," *> expression))
    <|> pure []

boolLit :: Parser Expr
boolLit =
        BoolLit True  <$ keyword "true"
    <|> BoolLit False <$ keyword "false"

number :: Parser Expr
number = token $ do
    whole    <- some (satisfy "digit" isDigit)
    fraction <- optional (char '.' *> some (satisfy "digit" isDigit))
    pure $ case fraction of
        Nothing   -> IntLit (read whole)
        Just frac -> RealLit (read (whole ++ "." ++ frac))

parentheses :: Parser Expr
parentheses = symbol "(" *> expression <* symbol ")"

-- Left-associative binary operator chaining
chainl1 :: Parser a -> Parser (a -> a -> a) -> Parser a
chainl1 p op = p >>= rest
  where
    rest x = (do f <- op; y <- p; rest (f x y)) <|> pure x

-- Statements ------------------------------------------------------------------

statement :: Parser Stmt
statement = ifStmt <|> forStmt <|> multiAssignStmt <|> assignStmt

multiAssignStmt :: Parser Stmt
multiAssignStmt = do
    names <- symbol "(" *> ((:) <$> identifier <*> many (symbol "," *> identifier)) <* symbol ")"
    void (symbol ":=")
    value <- expression
    void (symbol ";")
    pure (MultiAssign names value)

assignStmt :: Parser Stmt
assignStmt = do
    name  <- qualifiedName
    void (symbol ":=")
    value <- expression
    void (symbol ";")
    pure (Assign name value)

ifStmt :: Parser Stmt
ifStmt = do
    void (keyword "if")
    cond      <- expression
    void (keyword "then")
    thenStmts <- many statement
    elseStmts <- optional (keyword "else" *> many statement)
    void (keyword "end" *> keyword "if" *> symbol ";")
    pure (If cond thenStmts (maybe [] id elseStmts))

forStmt :: Parser Stmt
forStmt = do
    void (keyword "for")
    var   <- identifier
    void (keyword "in")
    lo    <- expression
    void (keyword "to")
    hi    <- expression
    mstep <- optional (keyword "step" *> expression)
    void (keyword "do")
    body  <- many statement
    void (keyword "end" *> keyword "for" *> symbol ";")
    pure (For var lo hi mstep body)

-- Methods, functions, blocks --------------------------------------------------

method :: Parser Method
method = do
    void (keyword "method")
    name       <- identifier
    void (keyword "algorithm")
    statements <- many statement
    void (keyword "end")
    endName    <- identifier
    void (symbol ";")
    when (name /= endName) empty
    pure (Method name statements)

funcDecl :: Parser Decl
funcDecl = ioDecl   -- only input/output inside functions

stringLit :: Parser String
stringLit = token $ do
    void (char '"')
    content <- many (satisfy "string char" (/= '"'))
    void (char '"')
    pure content

function :: Parser Func
function = do
    void (keyword "function")
    name  <- identifier
    decls <- many funcDecl
    void (keyword "external")
    void stringLit                       -- "C"
    void identifier                      -- external name (ignored for now)
    void (symbol "(")
    void (optional (identifier *> many (symbol "," *> identifier)))
    void (symbol ")" *> symbol ";")
    void (keyword "end")
    endName <- identifier
    void (symbol ";")
    when (name /= endName) empty
    pure (Func name decls name)

block :: Parser Block
block = do
    void (keyword "block")
    name         <- identifier
    declarations <- many decl
    funcs        <- many function
    void (optional (keyword "public"))
    methods      <- many method
    void (keyword "end")
    endName      <- identifier
    void (symbol ";")
    when (name /= endName) empty
    pure (Block name declarations funcs methods)

galecFile :: Parser Block
galecFile = spaces *> block <* spaces <* eof


--------------------------------------------------------------------------------
-- 6. ERROR FORMATTING
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
    in if n >= 1 && n <= length lns then lns !! (n - 1) else ""

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
-- 7. AST PRETTY-PRINTER  (read-only view of the tree)
--------------------------------------------------------------------------------

indent :: Int -> String -> String
indent n s = replicate (n * 4) ' ' ++ s

prettyDir :: Direction -> String
prettyDir Input  = "input"
prettyDir Output = "output"

prettyType :: Type -> String
prettyType Real    = "Real"
prettyType Integer = "Integer"
prettyType Boolean = "Boolean"
prettyType StringT = "String"

prettyAttr :: Attr -> String
prettyAttr (Attr name expr) = name ++ " = " ++ prettyExpr expr

prettyExpr :: Expr -> String
prettyExpr (Var n)     = n
prettyExpr (IntLit n)  = show n
prettyExpr (RealLit x) = show x
prettyExpr (BoolLit b) = show b
prettyExpr (Neg e)     = "-" ++ prettyExpr e
prettyExpr (Not e)     = "not " ++ prettyExpr e
prettyExpr (Add a b)   = prettyExpr a ++ " + "  ++ prettyExpr b
prettyExpr (Sub a b)   = prettyExpr a ++ " - "  ++ prettyExpr b
prettyExpr (Mul a b)   = prettyExpr a ++ " * "  ++ prettyExpr b
prettyExpr (Div a b)   = prettyExpr a ++ " / "  ++ prettyExpr b
prettyExpr (And a b)   = prettyExpr a ++ " and " ++ prettyExpr b
prettyExpr (Or  a b)   = prettyExpr a ++ " or "  ++ prettyExpr b
prettyExpr (Eq  a b)   = prettyExpr a ++ " = "  ++ prettyExpr b
prettyExpr (Ne  a b)   = prettyExpr a ++ " <> " ++ prettyExpr b
prettyExpr (Lt  a b)   = prettyExpr a ++ " < "  ++ prettyExpr b
prettyExpr (Le  a b)   = prettyExpr a ++ " <= " ++ prettyExpr b
prettyExpr (Gt  a b)   = prettyExpr a ++ " > "  ++ prettyExpr b
prettyExpr (Ge  a b)   = prettyExpr a ++ " >= " ++ prettyExpr b
prettyExpr (Call n as) = n ++ "(" ++ intercalate ", " (map prettyExpr as) ++ ")"

printAST :: Block -> IO ()
printAST b = putStrLn (treeBlock b)

treeBlock :: Block -> String
treeBlock (Block name decls funcs methods) =
    unlines $
        [ "Block: " ++ name, "" ]
        ++ section "-- Declarations --"        (concatMap treeDecl decls)
        ++ section "-- External Functions --"  (concatMap treeFunc funcs)
        ++ section "-- Methods --"             (concatMap treeMethod methods)
  where
    section title xs
        | null xs   = []
        | otherwise = ("  " ++ title) : xs ++ [""]

treeDecl :: Decl -> [String]
treeDecl (Decl dir ty name attrs) =
    [ "  " ++ prettyDir dir ++ " " ++ prettyType ty ++ " " ++ name
      ++ if null attrs then "" else "  (" ++ intercalate ", " (map prettyAttr attrs) ++ ")"
    ]
treeDecl (ParamDecl ty name)    = [ "  parameter " ++ prettyType ty ++ " " ++ name ]
treeDecl (StateDecl ty name)    = [ "  state "     ++ prettyType ty ++ " " ++ name ]
treeDecl (InternalDecl ty name) = [ "  "           ++ prettyType ty ++ " " ++ name ]

treeFunc :: Func -> [String]
treeFunc (Func name decls _) =
    [ "  function " ++ name ]
    ++ map ("    " ++) (concatMap treeDecl decls)
    ++ [ "    external \"C\"" ]

treeMethod :: Method -> [String]
treeMethod (Method name stmts) =
    [ "  method " ++ name ]
    ++ concatMap (treeStmt 2) stmts
    ++ [ "" ]

treeStmt :: Int -> Stmt -> [String]
treeStmt d (Assign name expr) =
    [ indent d "Assign", indent (d+1) ("Variable: " ++ name) ]
    ++ treeExpr (d+1) expr

treeStmt d (MultiAssign names expr) =
    [ indent d "MultiAssign", indent (d+1) ("Variables: " ++ intercalate ", " names) ]
    ++ treeExpr (d+1) expr

treeStmt d (If cond thenStmts elseStmts) =
    [ indent d "If", indent (d+1) "condition" ]
    ++ treeExpr (d+2) cond
    ++ [ indent (d+1) "then" ]
    ++ concatMap (treeStmt (d+2)) thenStmts
    ++ [ indent (d+1) "else" ]
    ++ concatMap (treeStmt (d+2)) elseStmts

treeStmt d (For var lo hi mstep body) =
    [ indent d ("For: " ++ var), indent (d+1) "from" ]
    ++ treeExpr (d+2) lo
    ++ [ indent (d+1) "to" ]
    ++ treeExpr (d+2) hi
    ++ stepTree (d+1) mstep
    ++ [ indent (d+1) "body" ]
    ++ concatMap (treeStmt (d+2)) body

stepTree :: Int -> Maybe Expr -> [String]
stepTree _ Nothing     = []
stepTree d (Just expr) = indent d "step" : treeExpr (d+1) expr

treeExpr :: Int -> Expr -> [String]
treeExpr d (Var name)    = [ indent d ("Variable: " ++ name) ]
treeExpr d (IntLit n)    = [ indent d ("Int: " ++ show n) ]
treeExpr d (RealLit x)   = [ indent d ("Real: " ++ show x) ]
treeExpr d (BoolLit b)   = [ indent d ("Boolean: " ++ show b) ]
treeExpr d (Neg e)       = indent d "Neg" : treeExpr (d+1) e
treeExpr d (Not e)       = indent d "Not" : treeExpr (d+1) e
treeExpr d (Add a b)     = binaryTree d "Add" a b
treeExpr d (Sub a b)     = binaryTree d "Sub" a b
treeExpr d (Mul a b)     = binaryTree d "Mul" a b
treeExpr d (Div a b)     = binaryTree d "Div" a b
treeExpr d (And a b)     = binaryTree d "And" a b
treeExpr d (Or  a b)     = binaryTree d "Or"  a b
treeExpr d (Eq  a b)     = binaryTree d "Eq"  a b
treeExpr d (Ne  a b)     = binaryTree d "Ne"  a b
treeExpr d (Lt  a b)     = binaryTree d "Lt"  a b
treeExpr d (Le  a b)     = binaryTree d "Le"  a b
treeExpr d (Gt  a b)     = binaryTree d "Gt"  a b
treeExpr d (Ge  a b)     = binaryTree d "Ge"  a b
treeExpr d (Call name args) =
    indent d ("Call: " ++ name) : concatMap (treeExpr (d+1)) args

binaryTree :: Int -> String -> Expr -> Expr -> [String]
binaryTree d name a b = indent d name : treeExpr (d+1) a ++ treeExpr (d+1) b


--------------------------------------------------------------------------------
-- 8. C CODE GENERATION  (monadic – Reader for context)
--------------------------------------------------------------------------------
--
-- Context carried by the CodeGen monad:
--   * prefix   – "self->" for struct members
--   * locals   – names that are not members (loop variables …)
--   * decls    – block declarations (needed for min/max clamps)
--
-- Indentation depth is still an explicit argument because it is purely
-- local to statement nesting and does not need to be in the environment.
--------------------------------------------------------------------------------

data GenEnv = GenEnv
    { genPrefix :: String
    , genLocals :: [String]
    , genDecls  :: [Decl]
    }

newtype CodeGen a = CodeGen { runCodeGen :: Reader GenEnv a }
    deriving (Functor, Applicative, Monad)

-- Run a CodeGen computation with an initial environment
evalCodeGen :: String -> [Decl] -> CodeGen a -> a
evalCodeGen pref decls m =
    runReader (runCodeGen m) (GenEnv pref [] decls)

-- Local helpers that query / modify the environment
askPrefix :: CodeGen String
askPrefix = CodeGen (asks genPrefix)

askLocals :: CodeGen [String]
askLocals = CodeGen (asks genLocals)

askDecls :: CodeGen [Decl]
askDecls = CodeGen (asks genDecls)

withLocal :: String -> CodeGen a -> CodeGen a
withLocal name (CodeGen m) =
    CodeGen $ local (\e -> e { genLocals = name : genLocals e }) m

-- Type mapping ----------------------------------------------------------------

cType :: Type -> String
cType Real    = "real32_T"
cType Integer = "int32_T"
cType Boolean = "boolean_T"
cType StringT = "const char*"

-- Declaration helpers ---------------------------------------------------------

declName :: Decl -> String
declName (Decl _ _ n _)     = n
declName (ParamDecl _ n)    = n
declName (StateDecl _ n)    = n
declName (InternalDecl _ n) = n

declCType :: Decl -> String
declCType (Decl _ t _ _)     = cType t
declCType (ParamDecl t _)    = cType t
declCType (StateDecl t _)    = cType t
declCType (InternalDecl t _) = cType t

isInput    (Decl Input  _ _ _) = True; isInput    _ = False
isOutput   (Decl Output _ _ _) = True; isOutput   _ = False
isParam    (ParamDecl _ _)     = True; isParam    _ = False
isState    (StateDecl _ _)     = True; isState    _ = False
isInternal (InternalDecl _ _)  = True; isInternal _ = False

-- Expression precedence -------------------------------------------------------

precOf :: Expr -> Int
precOf (Or  _ _) = 0
precOf (And _ _) = 1
precOf (Eq  _ _) = 2
precOf (Ne  _ _) = 2
precOf (Lt  _ _) = 2
precOf (Le  _ _) = 2
precOf (Gt  _ _) = 2
precOf (Ge  _ _) = 2
precOf (Add _ _) = 3
precOf (Sub _ _) = 3
precOf (Mul _ _) = 4
precOf (Div _ _) = 4
precOf (Neg _)   = 5
precOf (Not _)   = 5
precOf _         = 6

cRealLit :: Double -> String
cRealLit x = show x ++ "f"

-- Resolve a name to either a local or a struct member
resolveName :: String -> CodeGen String
resolveName n = do
    locals <- askLocals
    pref   <- askPrefix
    pure $ if n `elem` locals then n else pref ++ n

-- Core expression printer (monadic)
cExprPrec :: Int -> Bool -> Expr -> CodeGen String
cExprPrec parentPrec isRight e = do
    child <- cExpr e
    let need = precOf e < parentPrec
               || (isRight && precOf e == parentPrec && parentPrec `elem` [3,4])
    pure $ if need then "(" ++ child ++ ")" else child

cExpr :: Expr -> CodeGen String
cExpr (Var n)          = resolveName n
cExpr (IntLit n)       = pure (show n)
cExpr (RealLit x)      = pure (cRealLit x)
cExpr (BoolLit True)   = pure "TRUE"
cExpr (BoolLit False)  = pure "FALSE"
cExpr (Neg e)          = ("-" ++) <$> cExprPrec 5 False e
cExpr (Not e)          = ("!" ++) <$> cExprPrec 5 False e
cExpr (Add a b)        = binary "+" 3 a b
cExpr (Sub a b)        = binary "-" 3 a b
cExpr (Mul a b)        = binary "*" 4 a b
cExpr (Div a b)        = binary "/" 4 a b
cExpr (And a b)        = binary "&&" 1 a b
cExpr (Or  a b)        = binary "||" 0 a b
cExpr (Eq  a b)        = binary "==" 2 a b
cExpr (Ne  a b)        = binary "!=" 2 a b
cExpr (Lt  a b)        = binary "<"  2 a b
cExpr (Le  a b)        = binary "<=" 2 a b
cExpr (Gt  a b)        = binary ">"  2 a b
cExpr (Ge  a b)        = binary ">=" 2 a b
cExpr (Call "abs" [e]) = do
    arg <- cExpr e
    pure ("fabsf(" ++ arg ++ ")")
cExpr (Call "min" [a,b]) = do
    x <- cExpr a; y <- cExpr b
    pure ("fminf(" ++ x ++ ", " ++ y ++ ")")
cExpr (Call "max" [a,b]) = do
    x <- cExpr a; y <- cExpr b
    pure ("fmaxf(" ++ x ++ ", " ++ y ++ ")")
cExpr (Call n as) = do
    args <- mapM cExpr as
    pure (n ++ "(" ++ intercalate ", " args ++ ")")

binary :: String -> Int -> Expr -> Expr -> CodeGen String
binary op prec a b = do
    left  <- cExprPrec prec False a
    right <- cExprPrec prec True  b
    pure (left ++ " " ++ op ++ " " ++ right)

-- Look up min/max attributes --------------------------------------------------

lookupMinMax :: [Decl] -> String -> (Maybe Expr, Maybe Expr)
lookupMinMax decls name =
    let attrs = concat [ as | Decl _ _ n as <- decls, n == name ]
        findA key = case [e | Attr k e <- attrs, k == key] of
                        (e:_) -> Just e
                        []    -> Nothing
    in (findA "min", findA "max")

-- Emit a clamp assignment if the signal has both min and max
emitClamp :: Int -> String -> CodeGen [String]
emitClamp d name = do
    decls  <- askDecls
    lhs    <- resolveName name
    let (mMin, mMax) = lookupMinMax decls name
    case (mMin, mMax) of
        (Just lo, Just hi) -> do
            loS <- cExpr lo
            hiS <- cExpr hi
            pure [ indent d (lhs ++ " = EmbedSim_ClampValue(" ++ lhs ++ ", "
                             ++ loS ++ ", " ++ hiS ++ ");") ]
        _ -> pure []

-- Statement generator (monadic) -----------------------------------------------

cStmt :: Int -> Stmt -> CodeGen [String]
cStmt d (Assign name expr) = do
    decls <- askDecls
    lhs   <- resolveName name
    rhs0  <- cExpr expr
    let (mMin, mMax) = lookupMinMax decls name
    rhs <- case (mMin, mMax) of
             (Just lo, Just hi) -> do
                 loS <- cExpr lo
                 hiS <- cExpr hi
                 pure ("EmbedSim_ClampValue(" ++ rhs0 ++ ", " ++ loS ++ ", " ++ hiS ++ ")")
             _ -> pure rhs0
    pure [ indent d (lhs ++ " = " ++ rhs ++ ";") ]

-- Specialised multi-assign for known external FOC functions
cStmt d (MultiAssign names (Call "InvPark" args))
    | length args >= 3 && length names >= 2 = do
        vd    <- cExpr (args !! 0)
        vq    <- cExpr (args !! 1)
        theta <- cExpr (args !! 2)
        va    <- resolveName (names !! 0)
        vb    <- resolveName (names !! 1)
        clamps <- concat <$> mapM (emitClamp (d+1)) names
        pure $
            [ indent d "/* Inverse Park transform (EmbedSim) */"
            , indent d "{"
            , indent (d+1) "FocDq_T dqIn;"
            , indent (d+1) "FocAngle_T angleIn;"
            , indent (d+1) "FocAlphaBeta_T abOut;"
            , indent (d+1) ("dqIn.D = " ++ vd ++ ";")
            , indent (d+1) ("dqIn.Q = " ++ vq ++ ";")
            , indent (d+1) ("angleIn.ThetaE = " ++ theta ++ ";")
            , indent (d+1) "(void)InvPark_Transform_Matrix(&dqIn, &angleIn, &abOut);"
            , indent (d+1) (va ++ " = abOut.Alpha;")
            , indent (d+1) (vb ++ " = abOut.Beta;")
            ]
            ++ clamps
            ++ [ indent d "}" ]

cStmt d (MultiAssign names (Call "SVPWM" args))
    | length args >= 3 && length names >= 3 = do
        va  <- cExpr (args !! 0)
        vb  <- cExpr (args !! 1)
        vdc <- cExpr (args !! 2)
        da  <- resolveName (names !! 0)
        db  <- resolveName (names !! 1)
        dc  <- resolveName (names !! 2)
        pref <- askPrefix
        clamps <- concat <$> mapM (emitClamp (d+1)) names
        pure $
            [ indent d "/* Space Vector PWM (EmbedSim) */"
            , indent d "{"
            , indent (d+1) "FocAlphaBeta_T abIn;"
            , indent (d+1) "FocAngle_T angleIn;"
            , indent (d+1) "SVM_DutyCycle_T dutyOut;"
            , indent (d+1) ("abIn.Alpha = " ++ va ++ ";")
            , indent (d+1) ("abIn.Beta  = " ++ vb ++ ";")
            , indent (d+1) "/* Prefer self->theta when available (common in FOC controllers) */"
            , indent (d+1) ("angleIn.ThetaE = " ++ pref ++ "theta;")
            , indent (d+1) ("(void)SVM_CalculateDutyCycleFromAlphaBeta(&abIn, &angleIn, " ++ vdc ++ ", &dutyOut);")
            , indent (d+1) (da ++ " = dutyOut.Ta;")
            , indent (d+1) (db ++ " = dutyOut.Tb;")
            , indent (d+1) (dc ++ " = dutyOut.Tc;")
            ]
            ++ clamps
            ++ [ indent d "}" ]

cStmt d (MultiAssign names (Call fname args)) = do
    inArgs  <- mapM cExpr args
    outArgs <- mapM (\n -> ("&(" ++) . (++ ")") <$> resolveName n) names
    pure [ indent d (fname ++ "(" ++ intercalate ", " (inArgs ++ outArgs) ++ ");") ]

cStmt d (MultiAssign names expr) = do
    lhsList <- mapM resolveName names
    rhs     <- cExpr expr
    pure [ indent d "/* multi-assign fallback */"
         , indent d (intercalate " = " lhsList ++ " = " ++ rhs ++ ";")
         ]

cStmt d (If cond thenStmts elseStmts) = do
    condS   <- cExpr cond
    thenS   <- concat <$> mapM (cStmt (d+1)) thenStmts
    elseS   <- concat <$> mapM (cStmt (d+1)) elseStmts
    pure $
        [ indent d ("if (" ++ condS ++ ")")
        , indent d "{"
        ]
        ++ thenS
        ++ [ indent d "}"
           , indent d "else"
           , indent d "{"
           ]
        ++ elseS
        ++ [ indent d "}" ]

cStmt d (For var lo hi mstep body) = do
    loS   <- cExpr lo
    hiS   <- cExpr hi
    stepS <- maybe (pure "1") cExpr mstep
    bodyS <- withLocal var (concat <$> mapM (cStmt (d+1)) body)
    let init = "int32_T " ++ var ++ " = " ++ loS
        cond = var ++ " <= " ++ hiS
        incr = var ++ " += " ++ stepS
    pure $
        [ indent d ("for (" ++ init ++ "; " ++ cond ++ "; " ++ incr ++ ")")
        , indent d "{"
        ]
        ++ bodyS
        ++ [ indent d "}" ]

-- Header generation -----------------------------------------------------------

genHeader :: Block -> String
genHeader (Block name decls funcs _) =
    let guard     = map toUpperChar name ++ "_H_"
        inputs    = filter isInput decls
        outputs   = filter isOutput decls
        params    = filter isParam decls
        states    = filter isState decls
        internals = filter isInternal decls
        allFields = inputs ++ outputs ++ params ++ states ++ internals
        needsTransform = any (\(Func n _ _) -> n == "InvPark" || n == "SVPWM") funcs
    in unlines $
        [ "/**********************************************************************************************************************"
        , " * Generated from GALEC block: " ++ name
        , " * Target: EmbedSim (real32_T)"
        , " *********************************************************************************************************************/"
        , "#ifndef " ++ guard
        , "#define " ++ guard
        , ""
        , "#include \"embed_sim_sys_types.h\""
        ]
        ++ (if needsTransform
               then [ "#include \"embed_sim_foc_types.h\""
                    , "#include \"embed_sim_coordinate_transform.h\""
                    , "#include \"embed_sim_sv_pwm.h\""
                    ]
               else [])
        ++ [ ""
           , "typedef struct"
           , "{"
           ]
        ++ map (\d -> "    " ++ declCType d ++ " " ++ declName d ++ ";") allFields
        ++ [ "} " ++ name ++ ";"
           , ""
           , "/* Controller API */"
           , "void " ++ name ++ "_Startup(" ++ name ++ "* const self);"
           , "void " ++ name ++ "_DoStep(" ++ name ++ "* const self);"
           , ""
           , "#endif /* " ++ guard ++ " */"
           ]

-- Source generation -----------------------------------------------------------

genSource :: Block -> String
genSource (Block name decls _ methods) =
    let startup = findMethod "Startup" methods
        dostep  = findMethod "DoStep"  methods
        body method =
            case method of
                Just (Method _ stmts) ->
                    evalCodeGen "self->" decls (concat <$> mapM (cStmt 1) stmts)
                Nothing -> ["    /* empty */"]
    in unlines $
        [ "/**********************************************************************************************************************"
        , " * Generated from GALEC block: " ++ name
        , " * Target: EmbedSim (real32_T)"
        , " *********************************************************************************************************************/"
        , "#include \"" ++ name ++ ".h\""
        , ""
        , "void " ++ name ++ "_Startup(" ++ name ++ "* const self)"
        , "{"
        ]
        ++ body startup
        ++ [ "}"
           , ""
           , "void " ++ name ++ "_DoStep(" ++ name ++ "* const self)"
           , "{"
           ]
        ++ body dostep
        ++ [ "}"
           , ""
           ]

findMethod :: String -> [Method] -> Maybe Method
findMethod n ms = case filter (\(Method name _) -> name == n) ms of
                    (m:_) -> Just m
                    []    -> Nothing

toUpperChar :: Char -> Char
toUpperChar c | c >= 'a' && c <= 'z' = toEnum (fromEnum c - 32)
              | otherwise            = c

-- Write the two files
generateC :: Block -> IO ()
generateC block@(Block name _ _ _) = do
    let hFile = name ++ ".h"
        cFile = name ++ ".c"
    writeFile hFile (genHeader block)
    writeFile cFile (genSource block)
    putStrLn $ "Generated: " ++ hFile
    putStrLn $ "Generated: " ++ cFile


--------------------------------------------------------------------------------
-- 9. MAIN
--------------------------------------------------------------------------------

main :: IO ()
main = do
    args <- getArgs
    let file = case args of
                   (f:_) -> f
                   []    -> defaultSource
    source <- readFile file
    putStrLn $ "=== Parsing: " ++ file ++ " ==="
    putStrLn ""
    case runParser galecFile source 0 of
        Ok ast _ _   -> do
            printAST ast
            putStrLn ""
            putStrLn "=== Generating C code (EmbedSim / real32_T) ==="
            generateC ast
        Err msgs pos -> putStrLn (formatError source pos msgs)
