module Lib
    ( someFunc
    ) where

import Data.Matrix
-- import qualified Data.List.NonEmpty as NE (NonEmpty, zipWith, map)
import Data.Foldable

someFunc :: IO ()
someFunc = putStrLn "someFunc"

type NRInput = [Double]
type NROutput = [Double]
data NRSample = Sample { input :: NRInput, expected :: NROutput}
type NRMiniBatch = [NRSample]
data NRPass = Pass { wb :: Matrix Double }
data NRNetwork = NRNetwork [NRPass]

think :: NRNetwork -> NRInput -> NROutput
think = undefined

expectation :: (Foldable f) => f Double -> Double
expectation samples = (sum samples) / (fromIntegral $ length samples)

gradientFor :: NRNetwork -> NRSample -> [Double]
gradientFor = undefined 

approxGradient :: NRNetwork -> NRMiniBatch -> [Double]
approxGradient network miniBatchs = let gradientOfMinibatchs = map (gradientFor network) miniBatchs
                                        sum = foldr (zipWith (+)) (head gradientOfMinibatchs) (tail gradientOfMinibatchs)
                                    in map (/ fromIntegral (length gradientOfMinibatchs)) sum

aoeu :: NRMiniBatch -> NRNetwork -> NRNetwork
aoeu = undefined

train :: NRNetwork -> [NRMiniBatch] -> NRNetwork
train = foldr aoeu