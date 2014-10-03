SMOTE
=====

A WEKA compatible implementation of the SMOTE meta classification technique

SMOTE (Synthetic Minority Over-sampling TEchnique), is a method of dealing with class distribution skew in datasets designed by Chawla, Bowyer, Hall and Kegelmeyer[1]

In order to increase the presence of minority class examples in the problem the SMOTE process generates brand new minority class examples using the set of minority training examples as a base. It is somewhat similar to over-sampling of the minority class, except that entirely new, synthetic, examples are generated instead of only cloning existing ones.

Each SMOTE step works by selecting an existing minority example as a base and then selects, by way of a user defined parameter, a number of the example's nearest neighbors. From here, the original algorithm selects a neighbor at random. Finally for each attribute pair in the neighbor and base example a new synthetic attribute value is generated which falls between the two attribute values. One approach to generating each new attribute value is to calculate the difference in attribute values between the two examples, multiply it by a random number between 0 and 1, then add that value to the lower of the two existing attribute values.

1) N.V. Chawla, K.W. Bowyer, L.O. Hall, and W.P. Kegelmeyer. SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16(3):321-357, 2002
