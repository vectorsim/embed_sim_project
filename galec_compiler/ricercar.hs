{-# LANGUAGE RecordWildCards #-}
------------------------------------------------------------
-- Frankel's Ricercar: Flow and Commutator
-- Theme X, counter-theme Y, and their two orders.
-- Outputs frankel_ricercar.mp3 via ffmpeg + libmp3lame.
--
-- Lie Bracket : [X,Y] = -[Y,X] heard as melodic inversion.
-- Bracket XZ  : [X,Z] = -[Z,X]
-- Bracket YZ  : [Y,Z] = -[Z,Y]
-- Jacobi      : [X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0
--               heard as three cyclic brackets collapsing
--               onto a single unison — with a diminuendo,
--               so the vanishing has a dynamic shape too.
------------------------------------------------------------

module Main where

import Data.Int (Int16)
import Data.List (sortOn)
import Data.Word (Word16)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import System.Process
    ( CreateProcess(..)
    , StdStream(CreatePipe, Inherit)
    , createProcess
    , proc
    , waitForProcess
    )
import qualified System.IO as IO
import Text.Printf (printf)

------------------------------------------------------------
-- Time & pitch
------------------------------------------------------------

data Duration = W | H | Q | E | S
    deriving (Show, Eq)

ticks :: Duration -> Int
ticks W = 1920
ticks H = 960
ticks Q = 480
ticks E = 240
ticks S = 120

bar :: Int
bar = ticks W

newtype Pitch = Pitch Int
    deriving (Show, Eq, Ord)

c, cs, db, d, eb, e, f, fs, g, ab, a, bb, b :: Int -> Pitch
c  o = Pitch (12 * o)
cs o = Pitch (12 * o + 1)
db o = Pitch (12 * o + 1)
d  o = Pitch (12 * o + 2)
eb o = Pitch (12 * o + 3)
e  o = Pitch (12 * o + 4)
f  o = Pitch (12 * o + 5)
fs o = Pitch (12 * o + 6)
g  o = Pitch (12 * o + 7)
ab o = Pitch (12 * o + 8)
a  o = Pitch (12 * o + 9)
bb o = Pitch (12 * o + 10)
b  o = Pitch (12 * o + 11)

transposePitch :: Int -> Pitch -> Pitch
transposePitch n (Pitch p) = Pitch (p + n)

------------------------------------------------------------
-- Musical data
------------------------------------------------------------

data MusicElement = Note Pitch Duration | Rest Duration
    deriving (Show, Eq)

type Melody = [MusicElement]

note :: Pitch -> Duration -> MusicElement
note = Note

rest :: Duration -> MusicElement
rest = Rest

melodyTicks :: Melody -> Int
melodyTicks = sum . map go
  where
    go (Note _ d) = ticks d
    go (Rest d)   = ticks d

melodyBars :: Melody -> Double
melodyBars m = fromIntegral (melodyTicks m) / fromIntegral bar

fitsInBar :: Melody -> Bool
fitsInBar m = melodyTicks m `mod` ticks S == 0

------------------------------------------------------------
-- Transformations
------------------------------------------------------------

type Transformation = Melody -> Melody

transpose :: Int -> Transformation
transpose n = map go
  where
    go (Note p d) = Note (transposePitch n p) d
    go (Rest d)   = Rest d

-- Melodic inversion about a pitch axis — the musical analogue of -X.
-- invertAbout 48 mirrors around middle C (c 4).
invertAbout :: Int -> Melody -> Melody
invertAbout axis = map go
  where
    go (Note (Pitch n) d) = Note (Pitch (2 * axis - n)) d
    go (Rest d)           = Rest d

------------------------------------------------------------
-- The subject
------------------------------------------------------------

subject :: Melody
subject =
    [ note (c  4) S, note (b  3) S, note (c  4) E, note (g  3) E
    , note (ab 3) E, note (c  4) S, note (b  3) S, note (c  4) E
    , note (d  4) E
    , note (g  3) E, note (c  4) S, note (b  3) S, note (c  4) E
    , note (d  4) E
    , note (f  4) S, note (g  4) S, note (ab 4) Q
    , note (g  4) S, note (f  4) S, note (eb 4) E
    ]

answer :: Melody
answer = transpose 7 subject

subjectLow :: Melody
subjectLow = transpose (-12) subject

counterSubject :: Melody
counterSubject =
    [ note (g  4) E, note (f  4) E, note (eb 4) E, note (d  4) E
    , note (c  4) Q, note (d  4) E, note (eb 4) E
    , note (f  4) E, note (g  4) E, note (ab 4) E, note (g  4) E
    , note (f  4) E, note (eb 4) Q, note (d  4) E, note (c  4) E
    , note (d  4) Q
    ]

-- A third theme Z, used to witness the Jacobi identity
-- and to give the brackets [X,Z] and [Y,Z] a right-hand argument.
-- Exactly two bars, so it aligns with `subject`.
driftSubject :: Melody
driftSubject =
    [ note (e  4) Q, note (f  4) E, note (g  4) E
    , note (a  4) H
    , note (g  4) Q, note (f  4) E, note (e  4) E
    , note (d  4) H
    ]

------------------------------------------------------------
-- Music monad
------------------------------------------------------------

data Voice = Soprano | Alto | Bass
    deriving (Show, Eq)

data TimedEvent = TimedEvent
    { eventTime     :: Int
    , eventPitch    :: Pitch
    , eventDuration :: Int
    , eventVelocity :: Double
    , eventVoice    :: Voice
    } deriving (Show, Eq)

newtype Music a = Music
    { runMusic :: Voice -> Int -> (a, Int, [TimedEvent])
    }

instance Functor Music where
    fmap f (Music m) = Music $ \v t ->
        let (a, t', e) = m v t in (f a, t', e)

instance Applicative Music where
    pure x = Music $ \_ t -> (x, t, [])
    Music mf <*> Music mx = Music $ \v t ->
        let (f, t1, e1) = mf v t
            (x, t2, e2) = mx v t1
        in (f x, t2, e1 ++ e2)

instance Monad Music where
    Music m >>= f = Music $ \v t ->
        let (a, t1, e1) = m v t
            (b, t2, e2) = runMusic (f a) v t1
        in (b, t2, e1 ++ e2)

at :: Int -> Music a -> Music a
at dt (Music m) = Music $ \v t -> m v (t + dt)

withVoice :: Voice -> Music a -> Music a
withVoice v (Music m) = Music $ \_ t -> m v t

-- Scale the velocity of every event produced by a Music.
-- Used to shape dynamics: see `entryDyn` below.
withDynamic :: Double -> Music a -> Music a
withDynamic k (Music m) = Music $ \v t ->
    let (a, t', evs) = m v t
    in (a, t', map (\e -> e { eventVelocity = eventVelocity e * k }) evs)

parallel :: Music a -> Music b -> Music ()
parallel (Music ma) (Music mb) = Music $ \v t ->
    let (_, ea, eva) = ma v t
        (_, eb, evb) = mb v t
    in ((), max ea eb, eva ++ evb)

parallelMany :: [Music ()] -> Music ()
parallelMany = foldr parallel (pure ())

play :: Pitch -> Duration -> Music ()
play p d = Music $ \v t ->
    ((), t + ticks d, [TimedEvent t p (ticks d) 0.84 v])

pause :: Duration -> Music ()
pause d = Music $ \_ t -> ((), t + ticks d, [])

playMelody :: Melody -> Music ()
playMelody []     = pure ()
playMelody (x:xs) = case x of
    Note p d -> play p d >> playMelody xs
    Rest d   -> pause d  >> playMelody xs

------------------------------------------------------------
-- Form
------------------------------------------------------------

data SectionEntry = SectionEntry
    { seSection :: String
    , seVoice   :: Voice
    , seStart   :: Int
    , seMelody  :: Melody
    }

type SectionAList = [(String, [SectionEntry])]

sectionDefs :: SectionAList
sectionDefs =
    [ ( "Exposition"
      , [ SectionEntry "Exposition" Alto    0 subject
        , SectionEntry "Exposition" Soprano 2 answer
        , SectionEntry "Exposition" Bass    4 subjectLow
        ]
      )
    , ( "Flow X"
      , [ SectionEntry "Flow X" Soprano 0 subject
        , SectionEntry "Flow X" Alto    2 answer
        , SectionEntry "Flow X" Bass    4 subjectLow
        ]
      )
    , ( "Flow Y"
      , [ SectionEntry "Flow Y" Soprano 0 counterSubject
        , SectionEntry "Flow Y" Alto    1 (transpose (-5) counterSubject)
        , SectionEntry "Flow Y" Bass    2 (transpose (-12) counterSubject)
        ]
      )
    , ( "XY"
      , [ SectionEntry "XY" Soprano 0 subject
        , SectionEntry "XY" Alto    1 counterSubject
        , SectionEntry "XY" Bass    3 subjectLow
        ]
      )
    , ( "YX"
      , [ SectionEntry "YX" Bass    0 (transpose (-12) counterSubject)
        , SectionEntry "YX" Alto    1 answer
        , SectionEntry "YX" Soprano 3 subject
        ]
      )
    , ( "Lie Bracket"
      -- [X, Y] = -[Y, X].
      -- Bars 0–2: X then Y      (=   [X, Y])
      -- Bars 2–4: Y then X      (=   [Y, X])
      -- Bars 4–6: the whole thing mirrored
      --                         (=  -[Y, X] = [X, Y]).
      , [ SectionEntry "Lie Bracket" Alto    0 subject
        , SectionEntry "Lie Bracket" Bass    0 counterSubject
        , SectionEntry "Lie Bracket" Alto    2 counterSubject
        , SectionEntry "Lie Bracket" Bass    2 subjectLow
        , SectionEntry "Lie Bracket" Soprano 4 (invertAbout 48 subject)
        , SectionEntry "Lie Bracket" Alto    4 (invertAbout 48 counterSubject)
        , SectionEntry "Lie Bracket" Bass    4 (invertAbout 48 subjectLow)
        ]
      )
    , ( "Bracket XZ"
      -- [X, Z] = -[Z, X].
      -- Same shape as Lie Bracket, but with Z instead of Y.
      -- Alto: X, Z, inv X, inv Z.
      -- Bass: Z, X, inv Z, inv X.
      , [ SectionEntry "Bracket XZ" Alto    0 subject
        , SectionEntry "Bracket XZ" Bass    0 driftSubject
        , SectionEntry "Bracket XZ" Alto    2 driftSubject
        , SectionEntry "Bracket XZ" Bass    2 subject
        , SectionEntry "Bracket XZ" Alto    4 (invertAbout 48 subject)
        , SectionEntry "Bracket XZ" Bass    4 (invertAbout 48 driftSubject)
        , SectionEntry "Bracket XZ" Alto    6 (invertAbout 48 driftSubject)
        , SectionEntry "Bracket XZ" Bass    6 (invertAbout 48 subject)
        ]
      )
    , ( "Bracket YZ"
      -- [Y, Z] = -[Z, Y].
      -- Longer slots because counterSubject is 2.375 bars.
      , [ SectionEntry "Bracket YZ" Soprano 0 counterSubject
        , SectionEntry "Bracket YZ" Bass    0 driftSubject
        , SectionEntry "Bracket YZ" Soprano 3 driftSubject
        , SectionEntry "Bracket YZ" Bass    3 counterSubject
        , SectionEntry "Bracket YZ" Soprano 6 (invertAbout 48 counterSubject)
        , SectionEntry "Bracket YZ" Bass    6 (invertAbout 48 driftSubject)
        , SectionEntry "Bracket YZ" Soprano 9 (invertAbout 48 driftSubject)
        , SectionEntry "Bracket YZ" Bass    9 (invertAbout 48 counterSubject)
        ]
      )
    , ( "Jacobi"
      -- [X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0.
      -- Each voice cycles through a permutation of X, Y, Z,
      -- then all three collapse onto middle C in unison.
      -- The three final W's are at different dynamic levels
      -- (see `entryDyn`), so the vanishing has a dynamic shape.
      , [ SectionEntry "Jacobi" Soprano 0 subject
        , SectionEntry "Jacobi" Soprano 2 counterSubject
        , SectionEntry "Jacobi" Soprano 5 driftSubject
        , SectionEntry "Jacobi" Soprano 7 [ note (c 4) W ]

        , SectionEntry "Jacobi" Alto    0 counterSubject
        , SectionEntry "Jacobi" Alto    3 driftSubject
        , SectionEntry "Jacobi" Alto    5 subject
        , SectionEntry "Jacobi" Alto    7 [ note (c 4) W ]

        , SectionEntry "Jacobi" Bass    0 driftSubject
        , SectionEntry "Jacobi" Bass    2 subjectLow
        , SectionEntry "Jacobi" Bass    4 counterSubject
        , SectionEntry "Jacobi" Bass    7 [ note (c 4) W ]
        ]
      )
    , ( "Coda"
      , [ SectionEntry "Coda" Soprano 0 [ note (g 5) H, note (f 5) H, note (e 5) W ]
        , SectionEntry "Coda" Alto    0 [ note (eb 5) H, note (d 5) H, note (c 5) W ]
        , SectionEntry "Coda" Bass    0 [ note (c 3) H, note (g 2) H, note (c 3) W ]
        ]
      )
    ]

sectionLengthBars :: (String, [SectionEntry]) -> Double
sectionLengthBars (_, entries) =
    maximum $ 0 :
    [ fromIntegral (seStart e) + melodyBars (seMelody e) | e <- entries ]

sectionOffsets :: [Int]
sectionOffsets =
    let lens = map sectionLengthBars sectionDefs
        go _ []       = []
        go acc (l:ls) = acc : go (acc + ceiling l) ls
    in go 0 lens

-- Per-entry velocity scaling.
-- The three voices of the Jacobi section arrive on their final
-- unison at different dynamic levels, so the collapse is heard.
entryDyn :: SectionEntry -> Double
entryDyn e
    | seSection e == "Jacobi" && seStart e == 7 =
        case seVoice e of
            Soprano -> 0.40
            Alto    -> 0.65
            Bass    -> 1.00
    | otherwise = 1.0

fugue :: Music ()
fugue = parallelMany
    [ withVoice (seVoice e)
        (at ((abs0 + seStart e) * bar)
            (withDynamic (entryDyn e) (playMelody (seMelody e))))
    | (abs0, (_, entries)) <- zip sectionOffsets sectionDefs
    , e <- entries
    ]

fugueEvents :: [TimedEvent]
fugueEvents =
    let (_, _, evs) = runMusic fugue Soprano 0
    in sortOn eventTime evs

------------------------------------------------------------
-- Diagnostics
------------------------------------------------------------

checkVoicesFitInBar :: [SectionEntry] -> [String]
checkVoicesFitInBar =
    map format . filter (not . fitsInBar . seMelody)
  where
    format e =
        printf "  FAIL  %-18s %-8s start=%d  dur=%.3f bars  (ticks=%d)"
               (seSection e)
               (show (seVoice e))
               (seStart e)
               (melodyBars (seMelody e))
               (melodyTicks (seMelody e))

absoluteIntervals :: [(Double, Double, Voice, String)]
absoluteIntervals =
    concat
    [ [ ( fromIntegral abs0 + fromIntegral (seStart e)
        , fromIntegral abs0 + fromIntegral (seStart e) + melodyBars (seMelody e)
        , seVoice e
        , seSection e
        )
      | e <- entries
      ]
    | (abs0, (_, entries)) <- zip sectionOffsets sectionDefs
    ]

totalBars :: Int
totalBars =
    ceiling $ maximum $ 0 : [ end | (_, end, _, _) <- absoluteIntervals ]

barCoverage :: Double -> Double -> Int -> Double
barCoverage s e b =
    let bs = fromIntegral b
        be = bs + 1
        lo = max s bs
        hi = min e be
    in max 0 (hi - lo)

coverChar :: Double -> Char
coverChar c
    | c >= 0.75 = '#'
    | c >= 0.40 = '='
    | c >= 0.15 = '+'
    | c >  0    = ' '
    | otherwise = '.'

ganttRow :: Voice -> String
ganttRow v =
    [ coverChar (maximum $ 0 :
                 [ barCoverage s e b
                 | (s, e, v', _) <- absoluteIntervals, v' == v
                 ])
    | b <- [0 .. totalBars - 1]
    ]

sectionTag :: String -> Char
sectionTag name
    | name == "Exposition"  = 'E'
    | name == "Flow X"      = 'X'
    | name == "Flow Y"      = 'Y'
    | name == "XY"          = 'A'
    | name == "YX"          = 'B'
    | name == "Lie Bracket" = 'L'
    | name == "Bracket XZ"  = 'x'
    | name == "Bracket YZ"  = 'y'
    | name == "Jacobi"      = 'J'
    | name == "Coda"        = 'K'
    | otherwise             = '?'

sectionMarkerLine :: String
sectionMarkerLine =
    let marks = [ (abs0, sectionTag name)
                | (abs0, (name, _)) <- zip sectionOffsets sectionDefs
                ]
        go b = maybe ' ' id (lookup b marks)
    in map go [0 .. totalBars - 1]

barRuler :: (String, String)
barRuler =
    let digit n
            | n >= 0 && n <= 9 = toEnum (fromEnum '0' + n) :: Char
            | otherwise        = ' '
        tens  = [ if b `mod` 5 == 0 then digit (b `div` 10) else ' '
                | b <- [0 .. totalBars - 1] ]
        units = [ if b `mod` 5 == 0 then digit (b `mod` 10) else ' '
                | b <- [0 .. totalBars - 1] ]
    in (tens, units)

displaySections :: IO ()
displaySections = do
    putStrLn "Gantt timeline (one column = one bar)"
    putStrLn "====================================="
    putStrLn "  # full   = most   + half     little   . silent"
    putStrLn ""
    let (tens, units) = barRuler
        label w = printf "%-4s" w
    putStrLn $ label ""  ++ tens
    putStrLn $ label ""  ++ units
    putStrLn $ label ""  ++ replicate totalBars '-'
    putStrLn $ label "Sec" ++ sectionMarkerLine
    putStrLn $ label ""  ++ replicate totalBars '-'
    putStrLn $ label "Sop" ++ ganttRow Soprano
    putStrLn $ label "Alt" ++ ganttRow Alto
    putStrLn $ label "Bas" ++ ganttRow Bass
    putStrLn $ label ""  ++ replicate totalBars '-'
    putStrLn ""
    putStrLn "Section starts (absolute bar):"
    mapM_ (\(abs0, (name, _)) ->
              putStrLn $ printf "  %c  bar %2d  %s"
                                (sectionTag name) abs0 name)
          (zip sectionOffsets sectionDefs)
    putStrLn ""
    putStrLn "Bar-fit check (duration must be multiple of S = 120 ticks)"
    putStrLn "----------------------------------------------------------"
    let fails = concatMap (checkVoicesFitInBar . snd) sectionDefs
    if null fails
        then putStrLn "  All voices fit cleanly on bar subdivisions."
        else do
            putStrLn "  Some entries do not land on a clean subdivision:"
            mapM_ putStrLn fails
    putStrLn ""

------------------------------------------------------------
-- Humanization
------------------------------------------------------------

frac :: Double -> Double
frac x = x - fromIntegral (floor x :: Int)

jitter :: Int -> Double
jitter k = 2 * frac (fromIntegral k * 0.6180339887498949) - 1

humanize :: [TimedEvent] -> [TimedEvent]
humanize = zipWith go [0 :: Int ..]
  where
    go i e = e
        { eventTime     = max 0 (eventTime e + round (jitter i * 4))
        , eventVelocity = max 0.22 $
                          eventVelocity e * (1 + 0.08 * jitter (i + 7919))
        }

------------------------------------------------------------
-- Synthesis
------------------------------------------------------------

sampleRate :: Int
sampleRate = 32000

tempo :: Double
tempo = 78.0

secondsPerTick :: Double
secondsPerTick = 60.0 / (tempo * 480.0)

pitchFrequency :: Pitch -> Double
pitchFrequency (Pitch n) =
    440.0 * 2 ** ((fromIntegral (n + 12) - 69.0) / 12.0)

data Timbre = Timbre
    { timbreRolloff, timbreInharm, timbreDecay, timbreGain :: Double }

timbreOf :: Voice -> Timbre
timbreOf Soprano = Timbre 1.35 1.00015 1.15 0.105
timbreOf Alto    = Timbre 1.45 1.00020 1.00 0.115
timbreOf Bass    = Timbre 1.60 1.00030 0.75 0.130

panL, panR :: Voice -> Double
panL Soprano = 0.80; panL Alto = 0.52; panL Bass = 0.58
panR Soprano = 0.45; panR Alto = 0.78; panR Bass = 0.65

noteSample :: TimedEvent -> Int -> Double
noteSample ev i =
    let Timbre{..} = timbreOf (eventVoice ev)
        t   = fromIntegral i / fromIntegral sampleRate
        dur = fromIntegral (eventDuration ev) * secondsPerTick
        f   = pitchFrequency (eventPitch ev)
        attack  = min 1.0 (t / 0.0025)
        release = max 0.0 (min 1.0 ((dur - t) / 0.04))
        pitchDecay = timbreDecay * (0.55 + 0.45 * (f / 440.0) ** 0.4)
        env = attack * release * exp (-pitchDecay * t)
        knock =
            if t < 0.012
            then (1 - t/0.012) * 0.20
                 * sin (2*pi*(f*2.7)*t) * exp (-90*t)
            else 0
        harmonic n =
            let n' = fromIntegral n :: Double
                hF = f * n' * (timbreInharm ** (n'-1))
                am = 1 / (n' ** timbreRolloff)
                dc = exp (-(0.9 + hF/380) * t)
            in am * dc * sin (2*pi*hF*t)
        nPart = min 14 (max 1 (floor (0.48 * fromIntegral sampleRate / f)))
        harm  = sum [ harmonic n | n <- [1..nPart] ]
        reg   = case eventPitch ev of
                    Pitch n | n < 48 -> 1.20
                            | n < 60 -> 1.06
                            | otherwise -> 0.90
    in env * timbreGain * reg * eventVelocity ev * (harm + knock)

------------------------------------------------------------
-- Render
------------------------------------------------------------

stride :: Int -> [a] -> [a]
stride n = go
  where
    go []       = []
    go (y : ys) = y : go (drop (n - 1) ys)

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs =
    let (a, b) = splitAt n xs
    in a : chunksOf n b

renderAudio :: [TimedEvent] -> BL.ByteString
renderAudio events =
    let totalTicks = maximum (0 : [ eventTime e + eventDuration e | e <- events ])
        totalSamples = max 1 $ ceiling $
            fromIntegral totalTicks * secondsPerTick * fromIntegral sampleRate
        tickToSample t = floor $
            fromIntegral t * secondsPerTick * fromIntegral sampleRate

        prepared = sortOn (\(s, _, _) -> s)
            [ (tickToSample (eventTime e)
              ,tickToSample (eventTime e + eventDuration e)
              ,e)
            | e <- events ]

        sweep :: [(Int, Int, TimedEvent)] -> [(Int, Int, TimedEvent)]
              -> Int -> [(Double, Double)]
        sweep _ _ i | i >= totalSamples = []
        sweep pending active i =
            let (starting, pending') = span (\(s, _, _) -> s <= i) pending
                active' = [ a | a@(_, en, _) <- active ++ starting, en > i ]
                l = sum [ panL (eventVoice e) * noteSample e (i - s)
                        | (s, _, e) <- active' ]
                r = sum [ panR (eventVoice e) * noteSample e (i - s)
                        | (s, _, e) <- active' ]
            in (l, r) : sweep pending' active' (i + 1)

        samples = sweep prepared [] 0
        peak = maximum (0.001 :
                 [ max (abs l) (abs r)
                 | (l, r) <- stride 8 samples ])
        norm = 0.88 / peak

        to16 x = round (max (-1) (min 1 (tanh (x * norm * 1.05))) * 32767)
                 :: Int16
        le16 s = let w = fromIntegral s :: Word16
                 in [ fromIntegral (w `mod` 256)
                    , fromIntegral (w `div` 256) ]

        pcmChunks =
            [ BS.pack (concat [ le16 (to16 l) ++ le16 (to16 r)
                              | (l, r) <- chunk ])
            | chunk <- chunksOf 4096 samples ]
    in BL.fromChunks pcmChunks

writeMP3 :: FilePath -> BL.ByteString -> IO ()
writeMP3 path pcm = do
    let ff = proc "ffmpeg"
            [ "-y","-loglevel","error"
            , "-f","s16le","-ar",show sampleRate,"-ac","2"
            , "-i","pipe:0"
            , "-codec:a","libmp3lame","-q:a","4", path ]
    (Just hin,_,_,ph) <- createProcess ff
        { std_in = CreatePipe, std_out = Inherit, std_err = Inherit }
    BL.hPut hin pcm
    IO.hClose hin
    _ <- waitForProcess ph
    pure ()

writeMP4 :: FilePath -> BL.ByteString -> IO ()
writeMP4 path pcm = do
    let ff = proc "ffmpeg"
            [ "-y","-loglevel","error"
            -- Videospur: einfarbig dunkel, 1280x720, 2 fps (Standbild reicht)
            , "-f","lavfi","-i","color=c=0x14141e:s=1280x720:r=2"
            -- Audiospur: dasselbe rohe s16le wie bei writeMP3
            , "-f","s16le","-ar",show sampleRate,"-ac","2"
            , "-i","pipe:0"
            , "-c:v","libx264","-tune","stillimage","-pix_fmt","yuv420p"
            , "-c:a","aac","-b:a","192k"
            , "-shortest", path ]
    (Just hin,_,_,ph) <- createProcess ff
        { std_in = CreatePipe, std_out = Inherit, std_err = Inherit }
    BL.hPut hin pcm
    IO.hClose hin
    _ <- waitForProcess ph
    pure ()

------------------------------------------------------------
-- Main
------------------------------------------------------------

main :: IO ()
main = do
    putStrLn "Frankel's Ricercar: Flow and Lie Commutator"
    putStrLn "========================================="
    putStrLn ""
    putStrLn "Theme X        : subject"
    putStrLn "Counter-theme Y: counterSubject"
    putStrLn "Theme Z        : driftSubject"
    putStrLn ""
    putStrLn "Form:"
    putStrLn "  Exposition -> Flow X -> Flow Y -> XY -> YX"
    putStrLn "  -> Lie Bracket -> Bracket XZ -> Bracket YZ"
    putStrLn "  -> Jacobi -> Coda"
    putStrLn ""
    putStrLn "The two musical orders are heard separately"
    putStrLn "before the final contrapuntal combination."
    putStrLn ""
    putStrLn "Lie Bracket : [X,Y] = -[Y,X]  (melodic inversion)"
    putStrLn "Bracket XZ  : [X,Z] = -[Z,X]"
    putStrLn "Bracket YZ  : [Y,Z] = -[Z,Y]"
    putStrLn "Jacobi      : [X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0"
    putStrLn "              heard as three cyclic brackets"
    putStrLn "              converging on a unison, with a"
    putStrLn "              diminuendo as the sum vanishes."
    putStrLn ""

    displaySections

    let events = humanize fugueEvents
        pcm    = renderAudio events

    putStrLn $ "Events : " ++ show (length events)
    putStrLn "Writing frankel_ricercar.mp3"
    writeMP3 "frankel_ricercar.mp3" pcm
    putStrLn "Done -> frankel_ricercar.mp3"
    putStrLn "Writing frankel_ricercar.mp4"
    writeMP4 "frankel_ricercar.mp4" pcm
    putStrLn "Done -> frankel_ricercar.mp4"