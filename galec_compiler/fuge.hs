{-# LANGUAGE RecordWildCards #-}

------------------------------------------------------------
-- A Haskellian Self-Reflective Fugue
--
-- SUBJECT        = DATA
-- TRANSFORM      = FUNCTION
-- VOICES         = PARALLEL COMPOSITION
-- fix            = SELF-REFERENCE
--
-- The piece is built so that the *same* idea appears at three levels:
--
--   1. Musical subject  (the motif that is imitated)
--   2. Functional subject (transformations of that motif)
--   3. Meta subject     (fix generates a passage that
--                        refers to itself)
--
-- Listening to the self-echo section is hearing the fixed-point
-- equation   x = f(x)   made audible.
------------------------------------------------------------

module Main where

import Data.Int (Int16)
import Data.List (sortOn)
import Data.Word (Word16)
import qualified Data.ByteString.Lazy as BL
import System.Process
    ( CreateProcess(..)
    , StdStream(CreatePipe, Inherit)
    , createProcess
    , proc
    , waitForProcess
    )
import qualified System.IO as IO
import Data.Function (fix)


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


------------------------------------------------------------
-- Transformations (functions on data)
------------------------------------------------------------

type Transformation = Melody -> Melody

transpose :: Int -> Transformation
transpose n = map go
  where
    go (Note p d) = Note (transposePitch n p) d
    go (Rest d)   = Rest d


------------------------------------------------------------
-- The subject (data)
--
-- Opening cell inspired by BWV 847, kept short and clear
-- so the self-reference can be heard.
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


------------------------------------------------------------
-- The fixed point: self-reference made audible
--
--   selfEcho depth seed  =  fix (\self n -> ...) depth
--
-- Each deeper layer is the same seed, transformed.
-- The music literally contains a copy of itself.
------------------------------------------------------------

selfEcho :: Int -> Melody -> Melody
selfEcho maxDepth seed = fix go maxDepth
  where
    go :: (Int -> Melody) -> Int -> Melody
    go self n
        | n <= 0    = seed
        | otherwise =
            let deeper = self (n - 1)
                -- each deeper layer rises a fifth and is slightly longer
                layer  = transpose (7 * (maxDepth - n + 1)) deeper
            in seed
               ++ [rest S]
               ++ take (max 3 (length seed * 2 `div` 3)) layer
               ++ [rest E]
               ++ layer
               ++ [rest S]
               ++ take (max 2 (length seed `div` 2)) (transpose (-5) layer)

-- The seed is the head of the subject — the motif that
-- the whole fugue is “about”.
echoSeed :: Melody
echoSeed =
    [ note (c  4) S, note (b  3) S, note (c  4) E, note (g  3) E ]


------------------------------------------------------------
-- Music monad (parallel & sequential composition)
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

parallel :: Music a -> Music b -> Music ()
parallel (Music ma) (Music mb) = Music $ \v t ->
    let (_, ea, eva) = ma v t
        (_, eb, evb) = mb v t
    in ((), max ea eb, eva ++ evb)

parallelMany :: [Music ()] -> Music ()
parallelMany = foldr parallel (pure ())

sections :: [Music ()] -> Music ()
sections = foldr (>>) (pure ())

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

at' :: Voice -> Int -> Melody -> Music ()
at' v bars m = withVoice v $ at (bars * bar) (playMelody m)


------------------------------------------------------------
-- Form
--
-- The self-reflective centre is deliberate:
-- after the ordinary fugal argument, the music
-- turns around and listens to itself (fix).
------------------------------------------------------------

exposition :: Music ()
exposition = parallelMany
    [ at' Alto    0 subject
    , at' Soprano 2 answer
    , at' Bass    4 subjectLow
    ]

episode :: Music ()
episode = parallelMany
    [ at' Soprano 0 echoSeed
    , at' Alto    0 (transpose (-5) echoSeed)
    , at' Bass    1 (transpose (-12) echoSeed)
    , at' Soprano 2 (transpose 2 echoSeed)
    , at' Alto    2 (transpose (-3) echoSeed)
    ]

development :: Music ()
development = parallelMany
    [ at' Bass    0 subjectLow
    , at' Alto    2 answer
    , at' Soprano 4 subject
    ]

-- The mirror: the piece regards its own opening motif
-- through the fixed-point combinator.
selfReflection :: Music ()
selfReflection = parallelMany
    [ at' Soprano 0 (selfEcho 4 echoSeed)                 -- deepest mirror
    , at' Alto    1 (transpose (-5) (selfEcho 3 echoSeed))
    , at' Bass    2 (transpose (-12) (selfEcho 3 echoSeed))
    ]

counterpoint :: Music ()
counterpoint = parallelMany
    [ at' Soprano 0 counterSubject
    , at' Alto    0 (transpose (-5) counterSubject)
    , at' Bass    1 (transpose (-12) counterSubject)
    ]

-- Stretto: subject entries pile up tightly (the classic fugal climax)
finalEntries :: Music ()
finalEntries = parallelMany
    [ at' Bass    0 subjectLow
    , at' Alto    1 answer
    , at' Soprano 2 subject
    , at' Alto    3 (transpose (-5) counterSubject)
    , at' Bass    4 (transpose (-12) counterSubject)
    ]

coda :: Music ()
coda = parallelMany
    [ at' Soprano 0 [ note (g 5) H, note (f 5) H, note (e 5) W ]  -- Picardy
    , at' Alto    0 [ note (eb 5) H, note (d 5) H, note (c 5) W ]
    , at' Bass    0 [ note (c 3) H, note (g 2) H, note (c 3) W ]
    ]

-- The whole form is itself a value composed of functions.
fugue :: Music ()
fugue = sections
    [ exposition
    , episode
    , development
    , selfReflection      -- x = f(x)
    , counterpoint
    , finalEntries
    , coda
    ]

fugueEvents :: [TimedEvent]
fugueEvents =
    let (_, _, evs) = runMusic fugue Soprano 0
    in sortOn eventTime evs


------------------------------------------------------------
-- Light humanization
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
-- Piano-like synthesis (compact)
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

renderAudio :: [TimedEvent] -> BL.ByteString
renderAudio events =
    let totalTicks = maximum [ eventTime e + eventDuration e | e <- events ]
        totalSamples = ceiling (fromIntegral totalTicks * secondsPerTick
                                * fromIntegral sampleRate) :: Int
        tickToSample t = floor (fromIntegral t * secondsPerTick
                                * fromIntegral sampleRate)
        prepared =
            [ (tickToSample (eventTime e)
              ,tickToSample (eventTime e + eventDuration e)
              ,e)
            | e <- events ]
        sampleAt i =
            let active = [ (e, noteSample e (i-s))
                         | (s,en,e) <- prepared, i >= s, i < en ]
                l = sum [ panL (eventVoice e) * a | (e,a) <- active ]
                r = sum [ panR (eventVoice e) * a | (e,a) <- active ]
            in (l,r)
        peak = maximum $ 0.001 :
               [ max (abs l) (abs r)
               | i <- [0,32..totalSamples-1], let (l,r) = sampleAt i ]
        norm = 0.88 / peak
        to16 x = round (max (-1) (min 1 (tanh (x*norm*1.05))) * 32767) :: Int16
        le16 s = let w = fromIntegral s :: Word16
                 in [ fromIntegral (w `mod` 256), fromIntegral (w `div` 256) ]
        pcm = BL.pack $ concat
              [ le16 (to16 l) ++ le16 (to16 r)
              | i <- [0..totalSamples-1], let (l,r) = sampleAt i ]
    in pcm

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


------------------------------------------------------------
-- Main
------------------------------------------------------------

main :: IO ()
main = do
    putStrLn "A Haskellian Self-Reflective Fugue"
    putStrLn "=================================="
    putStrLn ""
    putStrLn "  subject   = data"
    putStrLn "  transform = function"
    putStrLn "  voices    = parallel composition"
    putStrLn "  fix       = self-reference"
    putStrLn ""
    putStrLn "Form:"
    putStrLn "  Exposition -> Episode -> Development"
    putStrLn "  -> Self-Reflection (fix, deeper mirror)"
    putStrLn "  -> Counterpoint -> Stretto -> Coda"
    putStrLn ""
    putStrLn "The middle section is the fixed-point equation"
    putStrLn "    x = f(x)"
    putStrLn "made audible: the motif contains a transformed"
    putStrLn "copy of itself."
    putStrLn ""

    let events = humanize fugueEvents
        pcm    = renderAudio events

    putStrLn $ "Events : " ++ show (length events)
    writeMP3 "self_reflective_fugue.mp3" pcm
    putStrLn "Done -> self_reflective_fugue.mp3"
