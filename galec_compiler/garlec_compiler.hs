

module Main where

import Data.Char (isSpace, isDigit, isAlpha, isAlphaNum)
import Data.List (intercalate, nub)
import Control.Applicative (Alternative(empty, (<|>)))
import System.Environment (getArgs)


defaultSource :: FilePath
defaultSource = "pmsm.galec"


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
    | MultiAssign [String] Expr
    | If  Expr [Stmt] [Stmt]
    | For String Expr Expr (Maybe Expr) [Stmt]
    deriving Show

data Method = Method String [Stmt]         deriving Show

data Func = Func String [Decl] String deriving Show  -- name, decls, external name

data Block  = Block String [Decl] [Func] [Method] deriving Show


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


attr :: Parser Attr
attr = do
    name <- identifier
    _    <- symbol "="
    val  <- expression
    return (Attr name val)


attrList :: Parser [Attr]
attrList = do
    _ <- symbol "("
    first <- attr
    rest  <- many (symbol "," *> attr)
    _ <- symbol ")"
    return (first : rest)


ioDecl :: Parser Decl
ioDecl = do
    d <- direction
    t <- dataType
    n <- identifier
    attrs <- optional attrList
    _ <- symbol ";"
    return (Decl d t n (maybe [] id attrs))


paramDecl :: Parser Decl
paramDecl = do
    _ <- keyword "parameter"
    t <- dataType
    n <- identifier
    _ <- symbol ";"
    return (ParamDecl t n)


stateDecl :: Parser Decl
stateDecl = do
    _ <- keyword "state"
    t <- dataType
    n <- identifier
    _ <- symbol ";"
    return (StateDecl t n)


internalDecl :: Parser Decl
internalDecl = do
    t <- dataType
    n <- identifier
    _ <- symbol ";"
    return (InternalDecl t n)


decl :: Parser Decl
decl = ioDecl <|> paramDecl <|> stateDecl <|> internalDecl


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
statement = ifStmt <|> forStmt <|> multiAssignStmt <|> assignStmt


multiAssignStmt :: Parser Stmt
multiAssignStmt = do
    _     <- symbol "("
    first <- identifier
    rest  <- many (symbol "," *> identifier)
    _     <- symbol ")"
    _     <- symbol ":="
    value <- expression
    _     <- symbol ";"
    return (MultiAssign (first : rest) value)


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


funcDecl :: Parser Decl
funcDecl = ioDecl  -- only input/output inside functions


stringLit :: Parser String
stringLit = token $ do
    _ <- char '"'
    content <- many (satisfy "string char" (/= '"'))
    _ <- char '"'
    return content


function :: Parser Func
function = do
    _      <- keyword "function"
    name   <- identifier
    decls  <- many funcDecl
    _      <- keyword "external"
    _      <- stringLit  -- "C"
    -- the call like InvPark(vd, vq, theta, v_alpha, v_beta);
    _      <- identifier  -- external name
    _      <- symbol "("
    _      <- optional (do
                _ <- identifier
                many (symbol "," *> identifier)
                return ())
    _      <- symbol ")"
    _      <- symbol ";"
    _      <- keyword "end"
    endName <- identifier
    _      <- symbol ";"
    if name == endName
        then return (Func name decls name)
        else empty


block :: Parser Block
block = do
    _            <- keyword "block"
    name         <- identifier
    declarations <- many decl
    funcs        <- many function
    _            <- optional (keyword "public")
    methods      <- many method
    _            <- keyword "end"
    endName      <- identifier
    _            <- symbol ";"
    if name == endName
        then return (Block name declarations funcs methods)
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
prettyExpr (Add a b)   = prettyExpr a ++ " + " ++ prettyExpr b
prettyExpr (Sub a b)   = prettyExpr a ++ " - " ++ prettyExpr b
prettyExpr (Mul a b)   = prettyExpr a ++ " * " ++ prettyExpr b
prettyExpr (Div a b)   = prettyExpr a ++ " / " ++ prettyExpr b
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
printAST block =
    putStrLn (treeBlock block)


treeBlock :: Block -> String
treeBlock (Block name decls funcs methods) =
    unlines $
        [ "Block: " ++ name
        , ""
        ]
        ++ (if null decls then [] else ["  -- Declarations --"] ++ concatMap treeDecl decls ++ [""])
        ++ (if null funcs then [] else ["  -- External Functions --"] ++ concatMap treeFunc funcs ++ [""])
        ++ (if null methods then [] else ["  -- Methods --"] ++ concatMap treeMethod methods)


treeDecl :: Decl -> [String]
treeDecl (Decl dir ty name attrs) =
    [ "  " ++ prettyDir dir ++ " " ++ prettyType ty ++ " " ++ name
      ++ if null attrs
           then ""
           else "  (" ++ intercalate ", " (map prettyAttr attrs) ++ ")"
    ]
treeDecl (ParamDecl ty name) =
    [ "  parameter " ++ prettyType ty ++ " " ++ name ]
treeDecl (StateDecl ty name) =
    [ "  state " ++ prettyType ty ++ " " ++ name ]
treeDecl (InternalDecl ty name) =
    [ "  " ++ prettyType ty ++ " " ++ name ]


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
    [ indent d "Assign"
    , indent (d + 1) ("Variable: " ++ name)
    ]
    ++ treeExpr (d + 1) expr


treeStmt d (MultiAssign names expr) =
    [ indent d "MultiAssign"
    , indent (d + 1) ("Variables: " ++ intercalate ", " names)
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
-- C Code Generation
--------------------------------------------------------------------------------
-- C Code Generation  (EmbedSim / real32_T target)
--------------------------------------------------------------------------------

cType :: Type -> String
cType Real    = "real32_T"
cType Integer = "int32_T"
cType Boolean = "boolean_T"
cType StringT = "const char*"

-- Collect field names from different decl kinds
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

isInput  (Decl Input  _ _ _) = True
isInput  _                   = False
isOutput (Decl Output _ _ _) = True
isOutput _                   = False
isParam  (ParamDecl _ _)     = True
isParam  _                   = False
isState  (StateDecl _ _)     = True
isState  _                   = False
isInternal (InternalDecl _ _) = True
isInternal _                  = False

-- Format a real literal with 'f' suffix for real32_T
cRealLit :: Double -> String
cRealLit x = show x ++ "f"

-- Precedence levels (higher = binds tighter)
-- 0: or, 1: and, 2: comparison, 3: add/sub, 4: mul/div, 5: unary, 6: atom
precOf :: Expr -> Int
precOf (Or _ _)  = 0
precOf (And _ _) = 1
precOf (Eq _ _)  = 2
precOf (Ne _ _)  = 2
precOf (Lt _ _)  = 2
precOf (Le _ _)  = 2
precOf (Gt _ _)  = 2
precOf (Ge _ _)  = 2
precOf (Add _ _) = 3
precOf (Sub _ _) = 3
precOf (Mul _ _) = 4
precOf (Div _ _) = 4
precOf (Neg _)   = 5
precOf (Not _)   = 5
precOf _         = 6   -- Var, lit, Call

-- Parenthesize child only when its precedence is lower than parent
-- (or equal for non-associative right side of Sub/Div)
parenIf :: Int -> Bool -> String -> [String] -> Expr -> String
parenIf parentPrec isRight pref locals e =
    let child = cExprPrec pref locals e
        need  = precOf e < parentPrec
                || (isRight && precOf e == parentPrec && parentPrec `elem` [3,4])
    in if need then "(" ++ child ++ ")" else child

-- Core expression printer with precedence
cExprPrec :: String -> [String] -> Expr -> String
cExprPrec pref locals (Var n)
    | n `elem` locals = n
    | otherwise       = pref ++ n
cExprPrec _ _ (IntLit n)      = show n
cExprPrec _ _ (RealLit x)     = cRealLit x
cExprPrec _ _ (BoolLit True)  = "TRUE"
cExprPrec _ _ (BoolLit False) = "FALSE"
cExprPrec pref locals (Neg e) =
    "-" ++ parenIf 5 False pref locals e
cExprPrec pref locals (Not e) =
    "!" ++ parenIf 5 False pref locals e
cExprPrec pref locals (Add a b) =
    parenIf 3 False pref locals a ++ " + " ++ parenIf 3 True pref locals b
cExprPrec pref locals (Sub a b) =
    parenIf 3 False pref locals a ++ " - " ++ parenIf 3 True pref locals b
cExprPrec pref locals (Mul a b) =
    parenIf 4 False pref locals a ++ " * " ++ parenIf 4 True pref locals b
cExprPrec pref locals (Div a b) =
    parenIf 4 False pref locals a ++ " / " ++ parenIf 4 True pref locals b
cExprPrec pref locals (And a b) =
    parenIf 1 False pref locals a ++ " && " ++ parenIf 1 True pref locals b
cExprPrec pref locals (Or a b) =
    parenIf 0 False pref locals a ++ " || " ++ parenIf 0 True pref locals b
cExprPrec pref locals (Eq a b) =
    parenIf 2 False pref locals a ++ " == " ++ parenIf 2 True pref locals b
cExprPrec pref locals (Ne a b) =
    parenIf 2 False pref locals a ++ " != " ++ parenIf 2 True pref locals b
cExprPrec pref locals (Lt a b) =
    parenIf 2 False pref locals a ++ " < "  ++ parenIf 2 True pref locals b
cExprPrec pref locals (Le a b) =
    parenIf 2 False pref locals a ++ " <= " ++ parenIf 2 True pref locals b
cExprPrec pref locals (Gt a b) =
    parenIf 2 False pref locals a ++ " > "  ++ parenIf 2 True pref locals b
cExprPrec pref locals (Ge a b) =
    parenIf 2 False pref locals a ++ " >= " ++ parenIf 2 True pref locals b
cExprPrec pref locals (Call "abs" [e]) =
    "fabsf(" ++ cExprPrec pref locals e ++ ")"
cExprPrec pref locals (Call "min" [a,b]) =
    "fminf(" ++ cExprPrec pref locals a ++ ", " ++ cExprPrec pref locals b ++ ")"
cExprPrec pref locals (Call "max" [a,b]) =
    "fmaxf(" ++ cExprPrec pref locals a ++ ", " ++ cExprPrec pref locals b ++ ")"
cExprPrec pref locals (Call n as) =
    n ++ "(" ++ intercalate ", " (map (cExprPrec pref locals) as) ++ ")"

-- Public entry (no surrounding parentheses needed)
cExprLocals :: String -> [String] -> Expr -> String
cExprLocals = cExprPrec

-- Look up min/max attributes for a signal name from the block decls
lookupMinMax :: [Decl] -> String -> (Maybe Expr, Maybe Expr)
lookupMinMax decls name =
    let attrsOf (Decl _ _ n attrs) | n == name = attrs
        attrsOf _ = []
        attrs = concatMap attrsOf decls
        findA key = case [e | Attr k e <- attrs, k == key] of
                        (e:_) -> Just e
                        []    -> Nothing
    in (findA "min", findA "max")


-- Emit optional EmbedSim_ClampValue for a signal that has min/max attributes
emitClamp :: String -> [String] -> [Decl] -> Int -> String -> [String]
emitClamp pref locals decls d name =
    let (mMin, mMax) = lookupMinMax decls name
        lhs = if name `elem` locals then name else pref ++ name
    in case (mMin, mMax) of
         (Just lo, Just hi) ->
             [ indent d (lhs ++ " = EmbedSim_ClampValue(" ++ lhs ++ ", "
                         ++ cExprLocals pref locals lo ++ ", "
                         ++ cExprLocals pref locals hi ++ ");") ]
         _ -> []

-- Statement generator
cStmt :: String -> [String] -> [Decl] -> Int -> Stmt -> [String]
cStmt pref locals decls d (Assign name expr) =
    let lhs = if name `elem` locals then name else pref ++ name
        (mMin, mMax) = lookupMinMax decls name
        rhs0 = cExprLocals pref locals expr
        -- If both min and max exist, wrap with EmbedSim_ClampValue
        rhs = case (mMin, mMax) of
                (Just lo, Just hi) ->
                    "EmbedSim_ClampValue(" ++ rhs0 ++ ", "
                    ++ cExprLocals pref locals lo ++ ", "
                    ++ cExprLocals pref locals hi ++ ")"
                _ -> rhs0
    in [ indent d (lhs ++ " = " ++ rhs ++ ";") ]

-- Special-case known external functions to call real EmbedSim APIs
cStmt pref locals decls d (MultiAssign names (Call "InvPark" args))
    | length args >= 3 && length names >= 2 =
        let vd    = cExprLocals pref locals (args !! 0)
            vq    = cExprLocals pref locals (args !! 1)
            theta = cExprLocals pref locals (args !! 2)
            va    = if (names !! 0) `elem` locals then names !! 0 else pref ++ (names !! 0)
            vb    = if (names !! 1) `elem` locals then names !! 1 else pref ++ (names !! 1)
        in [ indent d "/* Inverse Park transform (EmbedSim) */"
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
           ++ concatMap (\n -> emitClamp pref locals decls (d+1) n) names
           ++ [ indent d "}" ]

cStmt pref locals decls d (MultiAssign names (Call "SVPWM" args))
    | length args >= 3 && length names >= 3 =
        let va   = cExprLocals pref locals (args !! 0)
            vb   = cExprLocals pref locals (args !! 1)
            vdc  = cExprLocals pref locals (args !! 2)
            da   = if (names !! 0) `elem` locals then names !! 0 else pref ++ (names !! 0)
            db   = if (names !! 1) `elem` locals then names !! 1 else pref ++ (names !! 1)
            dc   = if (names !! 2) `elem` locals then names !! 2 else pref ++ (names !! 2)
        in [ indent d "/* Space Vector PWM (EmbedSim) */"
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
           ++ concatMap (\n -> emitClamp pref locals decls (d+1) n) names
           ++ [ indent d "}" ]

cStmt pref locals decls d (MultiAssign names (Call fname args)) =
    -- Generic fallback: trailing pointer outputs
    let inArgs  = map (cExprLocals pref locals) args
        outArgs = map (\n -> "&(" ++ (if n `elem` locals then n else pref ++ n) ++ ")") names
        allArgs = inArgs ++ outArgs
    in [ indent d (fname ++ "(" ++ intercalate ", " allArgs ++ ");") ]

cStmt pref locals decls d (MultiAssign names expr) =
    [ indent d "/* multi-assign fallback */"
    , indent d (intercalate " = " (map (\n -> if n `elem` locals then n else pref ++ n) names)
                ++ " = " ++ cExprLocals pref locals expr ++ ";")
    ]

cStmt pref locals decls d (If cond thenStmts elseStmts) =
    [ indent d ("if (" ++ cExprLocals pref locals cond ++ ")")
    , indent d "{"
    ]
    ++ concatMap (cStmt pref locals decls (d+1)) thenStmts
    ++ [ indent d "}"
       , indent d "else"
       , indent d "{"
       ]
    ++ concatMap (cStmt pref locals decls (d+1)) elseStmts
    ++ [ indent d "}" ]

cStmt pref locals decls d (For var lo hi mstep body) =
    let step = case mstep of
                 Just s  -> cExprLocals pref locals s
                 Nothing -> "1"
        init = "int32_T " ++ var ++ " = " ++ cExprLocals pref locals lo
        cond = var ++ " <= " ++ cExprLocals pref locals hi
        incr = var ++ " += " ++ step
        newLocals = var : locals
    in [ indent d ("for (" ++ init ++ "; " ++ cond ++ "; " ++ incr ++ ")")
       , indent d "{"
       ]
       ++ concatMap (cStmt pref newLocals decls (d+1)) body
       ++ [ indent d "}" ]

-- Generate header
genHeader :: Block -> String
genHeader (Block name decls funcs methods) =
    let guard = map toUpperChar name ++ "_H_"
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

-- Generate source
genSource :: Block -> String
genSource (Block name decls funcs methods) =
    let startup = findMethod "Startup" methods
        dostep  = findMethod "DoStep" methods
        pref    = "self->"
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
        ++ (case startup of
              Just (Method _ stmts) -> concatMap (cStmt pref [] decls 1) stmts
              Nothing               -> ["    /* empty */"])
        ++ [ "}"
           , ""
           , "void " ++ name ++ "_DoStep(" ++ name ++ "* const self)"
           , "{"
           ]
        ++ (case dostep of
              Just (Method _ stmts) -> concatMap (cStmt pref [] decls 1) stmts
              Nothing               -> ["    /* empty */"])
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

-- Write generated files
generateC :: Block -> IO ()
generateC block@(Block name _ _ _) = do
    let hFile = name ++ ".h"
        cFile = name ++ ".c"
    writeFile hFile (genHeader block)
    writeFile cFile (genSource block)
    putStrLn $ "Generated: " ++ hFile
    putStrLn $ "Generated: " ++ cFile


--------------------------------------------------------------------------------
-- Main
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