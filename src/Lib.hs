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

-- |Derivative of sigmoid.
sigmoid' :: Double -> Double
sigmoid' x = let s = sigmoid x
             in s * (1 - s)

feedforward :: NRInput-> NRNetwork -> NROutput 
feedforward = foldr activate

expectation :: (Foldable f) => f Double -> Double
expectation samples = (sum samples) / (fromIntegral $ length samples)

gradF :: NRNetwork -> NRInput -> NROutput -> [Double]
gradF [] a y = map (2*) (zipWith (-) a y)
gradF net a y = let (wb:net_) = net
                    t = affineT wb a
                    sigvec' = fromList $ map sigmoid' t
                    sigMat' = diagonal 0 sigvec'
                    jt = transpose $ sigMat' `multStd` (weights wb)
                    a_ = activate wb a
                    gF_ = gradF net_ a_ y
                in jt `linearT` gF_


-- |'a' is output dimension
gradOmega :: Int -> NRInput -> Matrix Double
gradOmega = fromLists .* replicate 

gradW :: NRNetwork -> NRInput -> NROutput -> Matrix Double
gradW net a y = let (wb:net_) = net
                    a_ = activate wb a
                    gO = gradOmega (length a_) a
                    gF_ = gradF net_ a_ y
                 in mapPos (\(i,j) x -> x * (gF_ !! i) ) gO

gradCx :: NRNetwork -> NRSample -> NRNetwork
gradCx [] _ = []
gradCx net Sample{input=a, expected=y} = let (wb:net_) = net
                                             a_ = activate wb a
                                             gW = gradW net a y
                                             gB = gradF net_ a_ y
                                         in  (Pass {weights=gW, biases=gB}:gradCx net_ (Sample a_ y) )

