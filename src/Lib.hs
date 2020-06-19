module Lib
    ( someFunc
    ) where

import Data.Matrix (multStd, Matrix, elementwise, colVector, transpose, diagonal, fromLists, mapPos)
-- import qualified Data.List.NonEmpty as NE (NonEmpty, zipWith, map)
import Data.Foldable
import Data.Vector (fromList)

someFunc :: IO ()
someFunc = putStrLn "someFunc"

type Vec = [Double]
type NRInput = Vec
type NROutput = Vec
data NRSample = Sample { input :: NRInput, expected :: NROutput}
type NRMiniBatch = [NRSample]
data NRPass = Pass { weights :: Matrix Double, biases :: [Double] }
type NRNetwork = [NRPass]
type DifferentialNetwork = NRNetwork

(.*) :: (c -> d) -> (a -> b -> c) -> (a -> b -> d)
(.*) = (.) . (.)

linearT :: Num f => Matrix f -> [f] -> [f]
linearT m l = toList $ multStd m (colVector (fromList l))

mAdd :: Num f => Matrix f -> Matrix f -> Matrix f
mAdd = elementwise (+)

vAdd :: Num f => [f] -> [f] -> [f]
vAdd = zipWith (+)

vSum :: Num f => [[f]] -> [f]
vSum (head:tail) = foldl vAdd head tail

affineT :: NRPass -> [Double] -> [Double]
affineT pass prevA = (linearT (weights pass) prevA) `vAdd` (biases pass)

sigmoid :: Double -> Double
sigmoid x = 1/(1+ exp (-x))

activate = (map sigmoid) .* affineT

-- derivative of sigmoid
sigmoid' :: Double -> Double
sigmoid' x = sigmoid x * (1 - sigmoid x)

feedforward :: NRInput-> NRNetwork -> NROutput 
feedforward = foldr activate

expectation :: (Foldable f) => f Double -> Double
expectation samples = (sum samples) / (fromIntegral $ length samples)

gradF :: NRNetwork -> NRInput -> NROutput -> [Double]
gradF [] a y = map (2*) (zipWith (-) a y)
gradF net a y = let wb = head net
                    t = affineT wb a
                    sigvec' = fromList $ map sigmoid' t
                    sigMat' = diagonal 0 sigvec'
                    jt = transpose $ sigMat' `multStd` (weights wb)
                    gF_ = gradF (tail net) (map sigmoid t) y
                in jt `linearT` gF_


-- |'a' is output dimension
gradOmega :: Int -> NRInput -> Matrix Double
gradOmega = fromLists .* replicate 

gradW :: NRNetwork -> NRInput -> NROutput -> Matrix Double
gradW net a y = let (wb:net_) = tail net
                    a_ = activate wb a
                    gO = gradOmega (length a_) a
                    gF_ = gradF net_ a_ y
                 in mapPos (\(i,j) x -> x * (gF_ !! i) ) gO

