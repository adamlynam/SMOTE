/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    SMOTE.java
 *    Copyright (C) 2008 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.meta;

import weka.classifiers.RandomizableSingleClassifierEnhancer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Randomizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.*;
import weka.core.AdditionalMeasureProducer;
import weka.core.EuclideanDistance;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

/**
 <!-- globalinfo-start -->
 * Class for bagging a classifier to reduce variance. Can do classification and regression depending on the base learner. <br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Leo Breiman (1996). SMOTE predictors. Machine Learning. 24(2):123-140.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Breiman1996,
 *    author = {Leo Breiman},
 *    journal = {Machine Learning},
 *    number = {2},
 *    pages = {123-140},
 *    title = {SMOTE predictors},
 *    volume = {24},
 *    year = {1996}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -P
 *  Size of each bag, as a percentage of the
 *  training set size. (default 100)</pre>
 * 
 * <pre> -O
 *  Calculate the out of bag error.</pre>
 * 
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 * 
 * <pre> -I &lt;num&gt;
 *  Number of iterations.
 *  (default 10)</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.trees.REPTree)</pre>
 * 
 * <pre> 
 * Options specific to classifier weka.classifiers.trees.REPTree:
 * </pre>
 * 
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf (default 2).</pre>
 * 
 * <pre> -V &lt;minimum variance for split&gt;
 *  Set minimum numeric class variance proportion
 *  of train variance for split (default 1e-3).</pre>
 * 
 * <pre> -N &lt;number of folds&gt;
 *  Number of folds for reduced error pruning (default 3).</pre>
 * 
 * <pre> -S &lt;seed&gt;
 *  Seed for random data shuffling (default 1).</pre>
 * 
 * <pre> -P
 *  No pruning.</pre>
 * 
 * <pre> -L
 *  Maximum tree depth (default -1, no maximum)</pre>
 * 
 <!-- options-end -->
 *
 * Options after -- are passed to the designated classifier.<p>
 *
 * @author Bernhard Pfahringer (bernhard@cs.waikato.ac.nz)
 * @version $Revision: 1.0 $
 */
public class SMOTE
  extends RandomizableSingleClassifierEnhancer 
  implements WeightedInstancesHandler, 
             TechnicalInformationHandler, AdditionalMeasureProducer {

  /** for serialization */
  static final long serialVersionUID = -505879962237199703L;
  
  /** The number of minority examples to generate, as a percentage of the number of provided minority examples */
  protected double m_MinoritySamplesToGenerate = 1.0;
  
  /** The number of majority example to sample, as a percentage of the number of provided majority examples */
  protected double m_MajoritySamplesToDraw = 1.0;
  
  /** The number of neighbors to consider during each synthetic example generation */
  protected int m_smoteNeighbors = 5;
  
  /** When set to true the SMOTE function will randomly select a neighbor per attribute instead of one per instance to generate a new synthetic instance */
  protected boolean m_neighborPerAttribute = false;
  
  /** When set to true the SMOTE function will check generated examples are not nearest neighbors to majority class examples */
  protected boolean m_syntheticExampleProtection = false;
  
  NominalToBinary binaryFilter = new NominalToBinary();
    
  /**
   * Constructor.
   */
  public SMOTE()
  {
    m_Classifier = new weka.classifiers.trees.J48();
    binaryFilter.setTransformAllValues(true);
  }
  
  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
 
    return "Classifier for generating synthetic minority examples to help reduce class skew. Can do classification "
      + "and regression depending on the base learner. \n\n"
      + "For more information, see\n\n"
      + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Leo Breiman");
    result.setValue(Field.YEAR, "1996");
    result.setValue(Field.TITLE, "UnderBag predictors");
    result.setValue(Field.JOURNAL, "Machine Learning");
    result.setValue(Field.VOLUME, "24");
    result.setValue(Field.NUMBER, "2");
    result.setValue(Field.PAGES, "123-140");
    
    return result;
  }

  /**
   * String describing default classifier.
   * 
   * @return the default classifier classname
   */
  protected String defaultClassifierString()
  {
    return "weka.classifiers.trees.REPTree";
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(5);

    newVector.addElement(new Option(
              "\tThe number of extra minority examples to generate,\n" 
              + "\tas a percentage of the number of provided minority examples (default 1.0)",
              "G", 1, "-G"));
    
    newVector.addElement(new Option(
              "\tThe number of majority examples to sample,\n" 
              + "\tas a percentage of the number of provided majority examples (default 1.0)",
              "M", 1, "-M"));

    newVector.addElement(new Option(
              "\tThe number of neighbors to consider during each\n"
              + "\tsynthetic example generation (default 5)",
              "k", 1, "-k"));
    
    newVector.addElement(new Option(
              "\tWhen set to true the SMOTE function will randomly select a neighbor per attribute\n" 
              + "\tinstead of one per instance to generate a new synthetic instance (default false)",
              "A", 0, "-A"));
    
    newVector.addElement(new Option(
              "\tWhen set to true the SMOTE function will check generated examples\n" 
              + "\tare not nearest neighbors to majority class examples (default false)",
              "P", 0, "-P"));

    Enumeration enu = super.listOptions();
    while (enu.hasMoreElements()) {
      newVector.addElement(enu.nextElement());
    }
    return newVector.elements();
  }


  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -P
   *  Size of each bag, as a percentage of the
   *  training set size. (default 100)</pre>
   * 
   * <pre> -O
   *  Calculate the out of bag error.</pre>
   * 
   * <pre> -S &lt;num&gt;
   *  Random number seed.
   *  (default 1)</pre>
   * 
   * <pre> -I &lt;num&gt;
   *  Number of iterations.
   *  (default 10)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   * <pre> -W
   *  Full name of base classifier.
   *  (default: weka.classifiers.trees.REPTree)</pre>
   * 
   * <pre> 
   * Options specific to classifier weka.classifiers.trees.REPTree:
   * </pre>
   * 
   * <pre> -M &lt;minimum number of instances&gt;
   *  Set minimum number of instances per leaf (default 2).</pre>
   * 
   * <pre> -V &lt;minimum variance for split&gt;
   *  Set minimum numeric class variance proportion
   *  of train variance for split (default 1e-3).</pre>
   * 
   * <pre> -N &lt;number of folds&gt;
   *  Number of folds for reduced error pruning (default 3).</pre>
   * 
   * <pre> -S &lt;seed&gt;
   *  Seed for random data shuffling (default 1).</pre>
   * 
   * <pre> -P
   *  No pruning.</pre>
   * 
   * <pre> -L
   *  Maximum tree depth (default -1, no maximum)</pre>
   * 
   <!-- options-end -->
   *
   * Options after -- are passed to the designated classifier.<p>
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    String minoritySamplesToGenerate = Utils.getOption('G', options);
    if (minoritySamplesToGenerate.length() != 0)
    {
      setMinoritySamplesToGenerate(Double.parseDouble(minoritySamplesToGenerate));
    }
    else
    {
      setMinoritySamplesToGenerate(1.0);
    }
    
    String majoritySamplesToDraw = Utils.getOption('M', options);
    if (majoritySamplesToDraw.length() != 0)
    {
        setMajoritySamplesToDraw(Double.parseDouble(majoritySamplesToDraw));
    }
    else
    {
        setMajoritySamplesToDraw(1.0);
    }

    String smoteNeighbors = Utils.getOption('k', options);
    if (smoteNeighbors.length() != 0)
    {
      setSmoteNeighbors(Integer.parseInt(smoteNeighbors));
    }
    else
    {
      setSmoteNeighbors(5);
    }
    
    setNeighborPerAttribute(Utils.getFlag('A', options));
    setSyntheticExampleProtection(Utils.getFlag('P', options));

    super.setOptions(options);
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions()
  {
    String [] superOptions = super.getOptions();
    String [] options = new String [superOptions.length + 8];

    int current = 0;
    options[current++] = "-G";
    options[current++] = "" + getMinoritySamplesToGenerate();
    options[current++] = "-M";
    options[current++] = "" + getMajoritySamplesToDraw();
    options[current++] = "-k";
    options[current++] = "" + getSmoteNeighbors();
    
    if (getNeighborPerAttribute())
        options[current++] = "-A";
    if (getSyntheticExampleProtection())
        options[current++] = "-P";

    System.arraycopy(superOptions, 0, options, current, 
		     superOptions.length);

    current += superOptions.length;
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String minoritySamplesToGenerateTipText()
  {
    return "The number of minority examples to generate, as a percentage of the number of provided minority examples.";
  }

  /**
   * Gets the number of minority examples to generate, as a percentage of the number of provided minority examples.
   *
   * @return the number of minority examples to generate, as a percentage.
   */
  public double getMinoritySamplesToGenerate()
  {
    return m_MinoritySamplesToGenerate;
  }
  
  /**
   * Sets the number of minority examples to generate, as a percentage of the number of provided minority examples.
   *
   * @param newMinoritySamplesToGenerate the number of minority examples to generate, as a percentage.
   */
  public void setMinoritySamplesToGenerate(double newMinoritySamplesToGenerate)
  {
    m_MinoritySamplesToGenerate = newMinoritySamplesToGenerate;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String majoritySamplesToDrawTipText()
  {
    return "The number of majority example to sample, as a percentage of the number of provided majority examples. (allows undersampling with SMOTE)";
  }

  /**
   * Gets the number of majority examples to draw, as a percentage of the number of provided majority examples.
   *
   * @return the number of majority examples to draw, as a percentage.
   */
  public double getMajoritySamplesToDraw()
  {
    return m_MajoritySamplesToDraw;
  }
  
  /**
   * Sets the number of majority examples to draw, as a percentage of the number of provided majority examples.
   *
   * @param newMajoritySamplesToDraw the number of majority examples to draw, as a percentage.
   */
  public void setMajoritySamplesToDraw(double newMajoritySamplesToDraw)
  {
    m_MajoritySamplesToDraw = newMajoritySamplesToDraw;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String smoteNeighborsTipText()
  {
    return "The number of neighbors to consider during each synthetic example generation.";
  }

  /**
   * Gets the number of neighbors to consider during each synthetic example generation.
   *
   * @return the number of neighbors to consider.
   */
  public int getSmoteNeighbors()
  {
    return m_smoteNeighbors;
  }
  
  /**
   * Sets the number of neighbors to consider during each synthetic example generation.
   *
   * @param newSmoteNeighbors the number of neighbors to consider.
   */
  public void setSmoteNeighbors(int newSmoteNeighbors)
  {
    m_smoteNeighbors = newSmoteNeighbors;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String neighborPerAttributeTipText()
  {
    return "When set to true the SMOTE function will randomly select a neighbor per attribute instead of one per instance to generate a new synthetic instance.";
  }

  /**
   * Gets if SMOTE is using a random neighbor per attribute to generate instances.
   *
   * @return true if neighbor per attribute is on.
   */
  public boolean getNeighborPerAttribute()
  {
    return m_neighborPerAttribute;
  }
  
  /**
   * Sets if SMOTE is using a random neighbor per attribute to generate instances.
   *
   * @param newNeighborPerAttribute set to true to turn neighbor per attribute on.
   */
  public void setNeighborPerAttribute(boolean newNeighborPerAttribute)
  {
    m_neighborPerAttribute = newNeighborPerAttribute;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String syntheticExampleProtectionTipText()
  {
    return "When set to true the SMOTE function will check generated examples are not nearest neighbors to majority class examples.";
  }

  /**
   * Gets if SMOTE is checking for majority class examples near generated instances.
   *
   * @return true if synthetic example protection is on.
   */
  public boolean getSyntheticExampleProtection()
  {
    return m_syntheticExampleProtection;
  }
  
  /**
   * Sets if SMOTE is checking for majority class examples near generated instances.
   *
   * @param newSyntheticExampleProtection set to true to turn synthetic example protection on.
   */
  public void setSyntheticExampleProtection(boolean newSyntheticExampleProtection)
  {
    m_syntheticExampleProtection = newSyntheticExampleProtection;
  }
  
  /**
   * SMOTE method.
   *
   * @param data the training data to be used for generating the
   * bagged classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  public void buildClassifier(Instances data) throws Exception
  {
    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    binaryFilter.setInputFormat(data);
    data = Filter.useFilter(data, binaryFilter);
    
    Instances minorityClassInstances = new Instances(data, 0);
    Instances majorityClassInstances = new Instances(data, 0);
    for(int i = 0; i < data.numInstances(); i++)
    {
      Instance instance = data.instance(i);
      if (instance.classValue() > 0.0)
      {
	majorityClassInstances.add(instance);
      }
      else
      {
	minorityClassInstances.add(instance);
      }
    }

    if (minorityClassInstances.numInstances() > majorityClassInstances.numInstances())
    {
      // swap to make minorityClassInstances the minority class
      Instances temp = minorityClassInstances;
      minorityClassInstances = majorityClassInstances;
      majorityClassInstances = temp;
    }

    int minoritySamplesToGenerate = (int)Math.round(minorityClassInstances.numInstances() * m_MinoritySamplesToGenerate); // number of minority class examples to generate
    
    Random random = new Random(m_Seed);
    
    Instances smotedData;
    if (m_MajoritySamplesToDraw == 1.0)
    {
        //add all the existing instances to the dataset
        smotedData = new Instances(data);
    }
    else
    {
        smotedData = new Instances(minorityClassInstances);
        
        int majoritySamplesToDraw = (int)Math.round(majorityClassInstances.numInstances() * m_MajoritySamplesToDraw);
        for (int i = 0; i < majoritySamplesToDraw; i++)
        {
            int nextMajorityInstance = random.nextInt(majorityClassInstances.numInstances());
            smotedData.add(majorityClassInstances.instance(nextMajorityInstance));
        }
    }
    
    LinearNNSearch minorityNeighborSearch = new LinearNNSearch(minorityClassInstances);
    minorityNeighborSearch.setDistanceFunction(new EuclideanDistance(minorityClassInstances));
    
    //generate the synthetic examples for the minority class
    for(int i = 0; i < minoritySamplesToGenerate; i++)
    {
        Instance randomMinorityInstance = minorityClassInstances.instance(random.nextInt(minorityClassInstances.numInstances()));
        Instances kNearestMinorityNeighbors = minorityNeighborSearch.kNearestNeighbours(randomMinorityInstance, m_smoteNeighbors);        
        Instance generatedInstance = smoteExampleGenerator(randomMinorityInstance, kNearestMinorityNeighbors, random);
        
        if (m_syntheticExampleProtection)
        {
            LinearNNSearch fullNeighborSearch = new LinearNNSearch(data);
            fullNeighborSearch.setDistanceFunction(new EuclideanDistance(data));
            Instances kFullNearestNeighbors = fullNeighborSearch.kNearestNeighbours(generatedInstance, 1);
            
            while (!syntheticExampleProtectionPassed(generatedInstance, kFullNearestNeighbors))
            {
                System.err.println("redo");
                randomMinorityInstance = minorityClassInstances.instance(random.nextInt(minorityClassInstances.numInstances()));
                kNearestMinorityNeighbors = minorityNeighborSearch.kNearestNeighbours(randomMinorityInstance, m_smoteNeighbors);
                generatedInstance = smoteExampleGenerator(randomMinorityInstance, kNearestMinorityNeighbors, random);
                kFullNearestNeighbors = fullNeighborSearch.kNearestNeighbours(generatedInstance, 1);
            }
                
            smotedData.add(generatedInstance);
        }
        else
        {
            smotedData.add(generatedInstance);
        }
    }

    if (m_Classifier instanceof Randomizable)
    {
        ((Randomizable) m_Classifier).setSeed(random.nextInt());
    }

    // build the classifier
    m_Classifier.buildClassifier(smotedData);
  }

  /**
   * Calculates the class membership probabilities for the given test
   * instance.
   *
   * @param instance the instance to be classified
   * @return preedicted class probability distribution
   * @throws Exception if distribution can't be computed successfully 
   */
  public double[] distributionForInstance(Instance instance) throws Exception
  {
    binaryFilter.input(instance);
    Instance filteredInstance = binaryFilter.output();
      
    double [] sums = new double [filteredInstance.numClasses()], newProbs;
    
    if (filteredInstance.classAttribute().isNumeric() == true)
    {
        sums[0] += m_Classifier.classifyInstance(filteredInstance);
    }
    else
    {
        newProbs = m_Classifier.distributionForInstance(filteredInstance);
        for (int j = 0; j < newProbs.length; j++)
        sums[j] += newProbs[j];
    }
    if (filteredInstance.classAttribute().isNumeric() == true || Utils.eq(Utils.sum(sums), 0))
    {
      return sums;
    }
    else
    {
      Utils.normalize(sums);
      return sums;
    }
  }
  
    public Enumeration enumerateMeasures()
    {
        if (m_Classifier instanceof AdditionalMeasureProducer)
            return ((AdditionalMeasureProducer)m_Classifier).enumerateMeasures();
        else
            return new Vector(0).elements();
    }

    public double getMeasure(String additionalMeasureName)
    {
        if (m_Classifier instanceof AdditionalMeasureProducer)
            return ((AdditionalMeasureProducer)m_Classifier).getMeasure(additionalMeasureName);
        else
            throw new IllegalArgumentException("Additional measures not supported by base classifier.");
    }

  /**
   * Returns description of the bagged classifier.
   *
   * @return description of the bagged classifier as a string
   */
  public String toString() {
    
    if (m_Classifier == null)
    {
      return "SMOTE: No model built yet.";
    }
    StringBuffer text = new StringBuffer();
    text.append("All the base classifiers: \n\n");
    text.append(m_Classifier.toString() + "\n\n");
    return text.toString();
  }

  public String getRevision() {
    //return RevisionUtils.extract("$Revision: 1.41 $");
    return "1.0";
  }
  
  /**
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String [] argv) {
    runClassifier(new SMOTE(), argv);
  }
  
  private Instance smoteExampleGenerator(Instance minorityInstance, Instances kNearestMinorityNeighbors, Random random)
  {
      Instance randomNeighbor = kNearestMinorityNeighbors.instance(random.nextInt(kNearestMinorityNeighbors.numInstances()));
      
      double syntheticValues[] = new double[minorityInstance.numAttributes()];
      for (int i = 0; i < minorityInstance.numAttributes(); i++)
      {
          if (m_neighborPerAttribute)
              randomNeighbor = kNearestMinorityNeighbors.instance(random.nextInt(kNearestMinorityNeighbors.numInstances()));
          
          if (Double.isNaN(minorityInstance.value(i)))
              syntheticValues[i] = randomNeighbor.value(i);
          else if (Double.isNaN(randomNeighbor.value(i)))
              syntheticValues[i] = minorityInstance.value(i);
          else
          {
              double newSyntheticValue = (randomNeighbor.value(i) - minorityInstance.value(i)) * random.nextDouble();

              if (minorityInstance.attribute(i).isNumeric())
                  syntheticValues[i] = minorityInstance.value(i) + newSyntheticValue;
              else if (minorityInstance.attribute(i).isNominal())
                  syntheticValues[i] = Math.round(minorityInstance.value(i) + newSyntheticValue);
              else
                  syntheticValues[i] = minorityInstance.value(i);
          }
      }
      
      return new Instance(1.0, syntheticValues);
  }
  
  private boolean syntheticExampleProtectionPassed(Instance instanceToCheck, Instances kNearestNeighbors)
  {
      for (int i = 0; i < kNearestNeighbors.numInstances(); i++)
          if (instanceToCheck.getClass() != kNearestNeighbors.instance(i).getClass())
              return false;
      return true;
  }
}