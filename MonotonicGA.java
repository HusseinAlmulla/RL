/**
 * Copyright (C) 2010-2017 Gordon Fraser, Andrea Arcuri and EvoSuite
 * contributors
 *
 * This file is part of EvoSuite.
 *
 * EvoSuite is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3.0 of the License, or
 * (at your option) any later version.
 *
 * EvoSuite is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with EvoSuite. If not, see <http://www.gnu.org/licenses/>.
 */
package org.evosuite.ga.metaheuristics;

import java.io.PrintWriter;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.descriptive.moment.Variance;
import org.evosuite.Properties;
import org.evosuite.Properties.Criterion;
import org.evosuite.TimeController;
import org.evosuite.assertion.AssertionGenerator;
import org.evosuite.assertion.SimpleMutationAssertionGenerator;
import org.evosuite.contracts.ContractChecker;
import org.evosuite.coverage.CoverageCriteriaAnalyzer;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.ConstructionFailedException;
import org.evosuite.ga.FitnessFunction;
import org.evosuite.ga.FitnessReplacementFunction;
import org.evosuite.ga.ReplacementFunction;
import org.evosuite.rmi.ClientServices;
import org.evosuite.rmi.service.ClientState;
import org.evosuite.testsuite.TestSuiteChromosome;
import org.evosuite.utils.LoggingUtils;
import org.evosuite.utils.Randomness;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



/**
 * Implementation of steady state GA
 * 
 * @author Gordon Fraser
 */
public class MonotonicGA<T extends Chromosome> extends GeneticAlgorithm<T> {

	private static final long serialVersionUID = 7846967347821123201L;

	protected ReplacementFunction replacementFunction;

	private final Logger logger = LoggerFactory.getLogger(MonotonicGA.class);

	/**
	 * Constructor
	 * 
	 * @param factory
	 *            a {@link org.evosuite.ga.ChromosomeFactory} object.
	 */
	public MonotonicGA(ChromosomeFactory<T> factory) {
		super(factory);

		setReplacementFunction(new FitnessReplacementFunction());
	}

	/**
	 * <p>
	 * keepOffspring
	 * </p>
	 * 
	 * @param parent1
	 *            a {@link org.evosuite.ga.Chromosome} object.
	 * @param parent2
	 *            a {@link org.evosuite.ga.Chromosome} object.
	 * @param offspring1
	 *            a {@link org.evosuite.ga.Chromosome} object.
	 * @param offspring2
	 *            a {@link org.evosuite.ga.Chromosome} object.
	 * @return a boolean.
	 */
	protected boolean keepOffspring(Chromosome parent1, Chromosome parent2, Chromosome offspring1,
			Chromosome offspring2) {
		return replacementFunction.keepOffspring(parent1, parent2, offspring1, offspring2);
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<T>();

		// Elitism
		logger.debug("Elitism");
		newGeneration.addAll(elitism());

		// Add random elements
		// new_generation.addAll(randomism());

		while (!isNextPopulationFull(newGeneration) && !isFinished()) {
			logger.debug("Generating offspring");

			T parent1 = selectionFunction.select(population);
			T parent2;
			if (Properties.HEADLESS_CHICKEN_TEST)
				parent2 = newRandomIndividual(); // crossover with new random
			// individual
			else
				parent2 = selectionFunction.select(population); // crossover
			// with existing
			// individual

			T offspring1 = (T) parent1.clone();
			T offspring2 = (T) parent2.clone();

			try {
				// Crossover
				if (Randomness.nextDouble() <= Properties.CROSSOVER_RATE) {
					crossoverFunction.crossOver(offspring1, offspring2);
				}

			} catch (ConstructionFailedException e) {
				logger.info("CrossOver failed");
				continue;
			}

			// Mutation
			notifyMutation(offspring1);
			offspring1.mutate();
			notifyMutation(offspring2);
			offspring2.mutate();

			if (offspring1.isChanged()) {
				offspring1.updateAge(currentIteration);
			}
			if (offspring2.isChanged()) {
				offspring2.updateAge(currentIteration);
			}



			// The two offspring replace the parents if and only if one of
			// the offspring is not worse than the best parent.
			for (FitnessFunction<T> fitnessFunction : fitnessFunctions) {
				fitnessFunction.getFitness(offspring1);
				notifyEvaluation(offspring1);
				fitnessFunction.getFitness(offspring2);
				notifyEvaluation(offspring2);

			}




			if (keepOffspring(parent1, parent2, offspring1, offspring2)) {
				logger.debug("Keeping offspring");

				// Reject offspring straight away if it's too long
				int rejected = 0;
				if (isTooLong(offspring1) || offspring1.size() == 0) {
					rejected++;
				} else {
					// if(Properties.ADAPTIVE_LOCAL_SEARCH ==
					// AdaptiveLocalSearchTarget.ALL)
					// applyAdaptiveLocalSearch(offspring1);
					newGeneration.add(offspring1);
				}

				if (isTooLong(offspring2) || offspring2.size() == 0) {
					rejected++;
				} else {
					// if(Properties.ADAPTIVE_LOCAL_SEARCH ==
					// AdaptiveLocalSearchTarget.ALL)
					// applyAdaptiveLocalSearch(offspring2);
					newGeneration.add(offspring2);
				}

				if (rejected == 1)
					newGeneration.add(Randomness.choice(parent1, parent2));
				else if (rejected == 2) {
					newGeneration.add(parent1);
					newGeneration.add(parent2);
				}
			} else {
				logger.debug("Keeping parents");
				newGeneration.add(parent1);
				newGeneration.add(parent2);
			}

		}

		population = newGeneration;
		// archive
		updateFitnessFunctionsAndValues();

		currentIteration++;
	}

	private T newRandomIndividual() {
		T randomChromosome = chromosomeFactory.getChromosome();
		for (FitnessFunction<?> fitnessFunction : this.fitnessFunctions) {
			randomChromosome.addFitness(fitnessFunction);
		}
		return randomChromosome;
	}

	/** {@inheritDoc} */
	@Override
	public void initializePopulation() {
		notifySearchStarted();
		currentIteration = 0;

		// Set up initial population
		generateInitialPopulation(Properties.POPULATION);
		logger.debug("Calculating fitness of initial population");
		calculateFitnessAndSortPopulation();

		this.notifyIteration();
	}

	private static final double DELTA = 0.000000001; // it seems there is some
	// rounding error in LS,
	// but hard to debug :(

	/** {@inheritDoc} */
	@Override
	public void generateSolution() {
		
		if (Properties.ENABLE_SECONDARY_OBJECTIVE_AFTER > 0 || Properties.ENABLE_SECONDARY_OBJECTIVE_STARVATION) {
			disableFirstSecondaryCriterion();
		}

		if (population.isEmpty()) {
			initializePopulation();
			assert!population.isEmpty() : "Could not create any test";
		}

		logger.debug("Starting evolution");
		int starvationCounter = 0;
		double bestFitness = Double.MAX_VALUE;
		double lastBestFitness = Double.MAX_VALUE;
		if (getFitnessFunction().isMaximizationFunction()) {
			bestFitness = 0.0;
			lastBestFitness = 0.0;
		}
		int counter=0;
		PrintWriter wrt;

		////////////////////
		int d = 84;	
		int numbers_of_selections[] = new int[84];
		double sums_of_rewards[] = new double[84];
		ArrayList<Integer> ad_selected = new ArrayList<Integer>();
		double total_reward = 0;
		int counter_i = 0;
		Map<Integer,Double> list = new TreeMap<Integer,Double>();
		/////////////////////
		
		//////// for TS
		NormalDistribution nd = new NormalDistribution();
		double mu[] = new double[84];
		Variance var = new Variance();
		double s[] = new double[84];
		double sigma [] = new double[84];
		double mean [] = new double[84];
        double varTotal = 0;
        double meanTotal = 0;

		
		
       /////////

		List<Integer> opt = new ArrayList<Integer>();
		
		for (int i=0; i<84; i++)
			opt.add(i);
		Random random = new Random();
		boolean stop_flag = false;
		int loop_ch =0;
		//try {
		//	wrt = new PrintWriter(new FileOutputStream( new File("D:\\evosiute-hint\\results.txt"), true));
		while (!isFinished() && !stop_flag) {


			//UCb
			int ad = 0;
//			double max_lower_bound = 0;
			double max_random = 0; // for TS
			//wrt.print("upper_bound ");
			
			//choose randomly from the options
			/*if (selection_flag) {
				int indx = random.nextInt(opt.size());
				ad = opt.get(indx);
				if (numbers_of_selections[ad] == 0) {					
					opt.remove(indx);
				}
				if (opt.size()==0)
					selection_flag = false;

			}
			else {*/
			
//				for (int j=0; j<d;j++) {
//					double upper_bound = 0;
//					if (numbers_of_selections[j] > 0) {
//						double average_reward = sums_of_rewards[j]/(double)numbers_of_selections[j];					
//						double delta_j = ((double)1/40) * Math.sqrt(Math.log((double)(counter_i))/(double)numbers_of_selections[j]);
//						upper_bound = average_reward + delta_j;
//					}
//					else
//						upper_bound = Double.MAX_VALUE;
//					if (upper_bound > max_lower_bound) {
//						max_lower_bound = upper_bound;
//						ad = j;					
//					}
//				}
				
				
			//}

//			ad_selected.add(ad);
//			numbers_of_selections[ad] = numbers_of_selections[ad] + 1;
			//End of UCB
			
			
			
			////// Start TS
			
				for (int j=0; j<d;j++) {
					double random_gauss = 0;
					if (numbers_of_selections[j] > 0) {		
						nd = new NormalDistribution(mu[j],s[j]);
						random_gauss = nd.sample();
					}
					else
						random_gauss = Double.MAX_VALUE;
					if (random_gauss > max_random) {
						max_random = random_gauss;
						ad = j;				
					}
				}
				
				
			//////End TS
				
			//stop when the selection reach good point.
//			if(counter_i > 100 && ((counter_i + loop_ch)% 20 == 0)) {
//				List<Integer> selection = Arrays.stream(numbers_of_selections).boxed().collect(Collectors.toList());
//				Collections.sort(selection,Collections.reverseOrder());
//				int diff = selection.get(0)-selection.get(1);
//				int div = selection.get(0)/selection.get(1);
//				
//				double ratio = ((double) selection.get(0)/(double) counter_i) * (double) 100;
//				if((diff > 20 || div >= 2) && (ratio >= 25)) {
//					stop_flag = true;
//					LoggingUtils.getEvoLogger().info("############# Stop: " + counter_i);
//				}
//			}

			removeFitnessFunction(getOneCriteria(ad));  // UCB and Thompson Sampling
			updateBestIndividualFromArchive();
			////////////////////////////////				


			logger.info("Population size before: " + population.size());
			// related to Properties.ENABLE_SECONDARY_OBJECTIVE_AFTER;
			// check the budget progress and activate a secondary criterion
			// according to the property value.


			{
				double bestFitnessBeforeEvolution = getBestFitness();
				evolve();
				sortPopulation();
				double bestFitnessAfterEvolution = getBestFitness();

				if (getFitnessFunction().isMaximizationFunction())
					assert(bestFitnessAfterEvolution >= (bestFitnessBeforeEvolution
							- DELTA)) : "best fitness before evolve()/sortPopulation() was: " + bestFitnessBeforeEvolution
					+ ", now best fitness is " + bestFitnessAfterEvolution;
					else
						assert(bestFitnessAfterEvolution <= (bestFitnessBeforeEvolution
								+ DELTA)) : "best fitness before evolve()/sortPopulation() was: " + bestFitnessBeforeEvolution
						+ ", now best fitness is " + bestFitnessAfterEvolution;
			}

			{
				double bestFitnessBeforeLocalSearch = getBestFitness();
				applyLocalSearch();
				double bestFitnessAfterLocalSearch = getBestFitness();

				if (getFitnessFunction().isMaximizationFunction())
					assert(bestFitnessAfterLocalSearch >= (bestFitnessBeforeLocalSearch
							- DELTA)) : "best fitness before applyLocalSearch() was: " + bestFitnessBeforeLocalSearch
					+ ", now best fitness is " + bestFitnessAfterLocalSearch;
					else
						assert(bestFitnessAfterLocalSearch <= (bestFitnessBeforeLocalSearch
								+ DELTA)) : "best fitness before applyLocalSearch() was: " + bestFitnessBeforeLocalSearch
						+ ", now best fitness is " + bestFitnessAfterLocalSearch;
			}

			/*
			 * TODO: before explanation: due to static state handling, LS can
			 * worse individuals. so, need to re-sort.
			 * 
			 * now: the system tests that were failing have no static state...
			 * so re-sorting does just hide the problem away, and reduce
			 * performance (likely significantly). it is definitively a bug
			 * somewhere...
			 */
			// sortPopulation();

			double newFitness = getBestFitness();

			if (getFitnessFunction().isMaximizationFunction())
				assert(newFitness >= (bestFitness - DELTA)) : "best fitness was: " + bestFitness
				+ ", now best fitness is " + newFitness;
				else
					assert(newFitness <= (bestFitness + DELTA)) : "best fitness was: " + bestFitness
					+ ", now best fitness is " + newFitness;
					bestFitness = newFitness;

					if (Double.compare(bestFitness, lastBestFitness) == 0) {
						starvationCounter++;
					} else {
						logger.info("reset starvationCounter after " + starvationCounter + " iterations");
						starvationCounter = 0;
						lastBestFitness = bestFitness;

					}

					updateSecondaryCriterion(starvationCounter);

					logger.info("Current iteration: " + currentIteration);
					this.notifyIteration();

					logger.info("Population size: " + population.size());
					logger.info("Best individual has fitness: " + population.get(0).getFitness());
					logger.info("Worst individual has fitness: " + population.get(population.size() - 1).getFitness());

					////////////////////
					/*updateBestIndividualFromArchive();
					TestSuiteChromosome testSuite = (TestSuiteChromosome) this.getBestIndividual();
					TestSuiteGenerator ts = new TestSuiteGenerator();
					double coverage = CoverageCriteriaAnalyzer.analyzeCoverageNew(testSuite, Criterion.BRANCH);//ts.posttest(testSuite);
					LoggingUtils.getEvoLogger().info("coverage is :" + coverage);*/
					
					updateBestIndividualFromArchive();
					TestSuiteChromosome testSuite = (TestSuiteChromosome) this.getBestIndividual();
					
					double reward_score=0.0;

					if(Properties.REWARD == Properties.Reward.COVERAGE) {						
						reward_score = CoverageCriteriaAnalyzer.analyzeCoverageNew(testSuite, Criterion.BRANCH);
						
					}
					else if (Properties.REWARD == Properties.Reward.MUTATION) {	
//						AssertionGenerator asserter = new SimpleMutationAssertionGenerator();
//						reward_score = asserter.addAssertionsNew(testSuite);
						reward_score = CoverageCriteriaAnalyzer.analyzeCoverageNew(testSuite, Criterion.STRONGMUTATION);
					}
					
					LoggingUtils.getEvoLogger().info(" reward_score " + NumberFormat.getPercentInstance().format(reward_score));
					//UCB
					// sorting the first round 
					/*if(counter_i < 84) {
						list.put(counter_i, reward_score);
					}
					else {
						for(int i=0;i<list.size(); i++)
							LoggingUtils.getEvoLogger().info("    ["+i+"] : "+ list.get(i));
						
						
						//sorting the list of reward and print them.
						Map<Integer, Double> sortedMap = list.entrySet().stream().sorted(Entry.comparingByValue()).collect(Collectors.toMap(Entry::getKey, Entry::getValue,(e1, e2) -> e1, LinkedHashMap::new));
						Set<Integer> keys =  sortedMap.keySet();
						Iterator<Integer> it = keys.iterator();
						while(it.hasNext()) {
							int t= it.next();
							System.out.println(sortedMap.get(t));
						}					
							
					}*/
					sums_of_rewards[ad] = sums_of_rewards[ad] + reward_score;
					total_reward = total_reward + reward_score;
					counter_i++;
					//End of UCB
					
					///// TS
					ad_selected.add(ad);
					numbers_of_selections[ad] = numbers_of_selections[ad] + 1;
					sums_of_rewards[ad] = sums_of_rewards[ad] + reward_score;
					
					// calculate the variance for all reward
					double newMeanTotal = (meanTotal * (counter_i - 1) + reward_score)/counter_i;
					varTotal = ((counter_i -1) * varTotal + (reward_score - newMeanTotal) * (reward_score - meanTotal)) / counter_i;
					meanTotal = newMeanTotal;
					if (counter_i < 84) { // first round the mu=reward and the s=1
						mu[ad] = reward_score;
						s[ad] = 1;
					}
					else {
						int ns = numbers_of_selections[ad];
						//sigma[ad] =  ((ns - 1) * sigma[ad] + (reward_score - newMean) * (reward_score - mu[ad])) / ns;
						
						mu[ad] = ((mu[ad] * varTotal) + (reward_score * s[ad])) / (varTotal + s[ad]);
					    s[ad] = (s[ad] * varTotal) / (s[ad] * varTotal);					 
					}
					///// TS

					// printing each criterion with coverage score
					/*		wrt.println("Counter : " + counter++ + "  score: " + coverage_score +"\n");
						wrt.print("         [");
						for(Criterion c : Properties.CRITERION)
							wrt.print(" : "+ c.name());
 						wrt.println("]\n");
					 */
					///////////////////
					
					
		}			

		///////////////////

		// get the max value in the sum_of_rewards array
		int max = 0;			
		for(int i=0; i < sums_of_rewards.length; i++) {
			if(sums_of_rewards[i] > sums_of_rewards[max]) {
				max = i;
			}
		}
		//wrt.println(" Max List: ");
		/// new editing
		TestSuiteChromosome testSuite = (TestSuiteChromosome) this.getBestIndividual();
		ContractChecker.setActive(false);
		ClientServices.getInstance().getClientNode().changeState(ClientState.EVALUATION);
		AssertionGenerator asserter = new SimpleMutationAssertionGenerator();
		

		// calculate the initial max reward value depending on the value of max
		removeFitnessFunction(getOneCriteria(max));
		double max_reward = asserter.addAssertionsNew(testSuite);
		//wrt.print(" reward : "+ max + " is " + max_reward);

		// /*check if there are more than one sum_of_reward have same value of the max
		 //* if there are then check which one of these values have the highest mutation score 
		 //* these used to break the tie.*/
		 
		for(int i=0; i < sums_of_rewards.length; i++) {
			if(sums_of_rewards[i] == sums_of_rewards[max]) {
				//// new editing
				// to avoid calculation the initial max reward twice
				if (i == max)
					continue;
				removeFitnessFunction(getOneCriteria(i));
				testSuite = (TestSuiteChromosome) this.getBestIndividual();
				double new_ewward = asserter.addAssertionsNew(testSuite);					
				if (new_ewward > max_reward) {
					max = i;
					max_reward = new_ewward;
				}
				//	wrt.print("  ,   reward : [" + i + "] is " + new_ewward);
			}
		}

		String strg= "";
		for(int j=0; j< getOneCriteria(max).length; j++) {
		       strg +=getOneCriteria(max)[j].toString();
		       if(j < getOneCriteria(max).length-1)
		    	   strg+=":";
		}

		LoggingUtils.getEvoLogger().info("The best option is =" + strg);

		removeFitnessFunction(getOneCriteria(max));
		String crit[] = new String[84]; 
		for(int i=0;i<sums_of_rewards.length; i++) {
			String st = "";
			for(int j=0; j< getOneCriteria(i).length; j++) {
				 st +=getOneCriteria(i)[j].toString();
				 if(j < getOneCriteria(max).length-1)
					 	st+=":";
			}
			crit[i] = st;
		}
		
		Map<Integer, Double> sortedMap = list.entrySet().stream().sorted(Entry.comparingByValue()).collect(Collectors.toMap(Entry::getKey, Entry::getValue,(e1, e2) -> e1, LinkedHashMap::new));
		Set<Integer> keys =  sortedMap.keySet();
		Iterator<Integer> it = keys.iterator();
		LoggingUtils.getEvoLogger().info(" Sorted list of reward: ");
		while(it.hasNext()) {
			int t= it.next();
			LoggingUtils.getEvoLogger().info("    ["+crit[t]+ "] : "+ sortedMap.get(t));
		}			
		
		LoggingUtils.getEvoLogger().info(" sums_of_rewards using " + Properties.REWARD +" : ");
		for(int i=0;i<sums_of_rewards.length; i++)
			LoggingUtils.getEvoLogger().info("    ["+crit[i]+ "] : "+ sums_of_rewards[i]);
		
		LoggingUtils.getEvoLogger().info(" numbers_of_selections: ");
		for(int i=0;i<numbers_of_selections.length; i++)
			LoggingUtils.getEvoLogger().info("    ["+crit[i]+"] : "+ numbers_of_selections[i]);
		


		/*wrt.println(" ");

			wrt.println(" sums_of_rewards: ");
			for(int i=0;i<sums_of_rewards.length; i++)
				wrt.print("  ,  ["+i + "] : "+ sums_of_rewards[i]);
			wrt.println("\n");
			wrt.println(" numbers_of_selections: ");
			for(int i=0;i<numbers_of_selections.length; i++)
				wrt.print("  ,  ["+i+"] : "+ numbers_of_selections[i]);
			wrt.println("\n\n\n\n\n");
			wrt.close();
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}*/
		TimeController.execute(this::updateBestIndividualFromArchive, "update from archive", 5_000);
		notifySearchFinished();

		/////////////////////////
		// archive

		///////////////////////////////////////////////////////////////////////////////////////////////////////
		//This is the second round of search 
		//////////////////////////////////////////////////////////////		
		/*	notifySearchStarted();

		starvationCounter = 0;
		bestFitness = Double.MAX_VALUE;
		lastBestFitness = Double.MAX_VALUE;
		if (getFitnessFunction().isMaximizationFunction()) {
			bestFitness = 0.0;
			lastBestFitness = 0.0;
		}
		LoggingUtils.getEvoLogger().info("\n* Post Search start after the main search finished ------ ");

		ClientServices.getInstance().getClientNode().changeState(ClientState.POSTSEARCH);
			while (!isFinished()) {				

				logger.info("Population size before: " + population.size());

				{
					double bestFitnessBeforeEvolution = getBestFitness();
					evolve();
					sortPopulation();
					double bestFitnessAfterEvolution = getBestFitness();

					if (getFitnessFunction().isMaximizationFunction())
						assert(bestFitnessAfterEvolution >= (bestFitnessBeforeEvolution
								- DELTA)) : "best fitness before evolve()/sortPopulation() was: " + bestFitnessBeforeEvolution
						+ ", now best fitness is " + bestFitnessAfterEvolution;
						else
							assert(bestFitnessAfterEvolution <= (bestFitnessBeforeEvolution
									+ DELTA)) : "best fitness before evolve()/sortPopulation() was: " + bestFitnessBeforeEvolution
							+ ", now best fitness is " + bestFitnessAfterEvolution;
				}

				{
					double bestFitnessBeforeLocalSearch = getBestFitness();
					applyLocalSearch();
					double bestFitnessAfterLocalSearch = getBestFitness();

					if (getFitnessFunction().isMaximizationFunction())
						assert(bestFitnessAfterLocalSearch >= (bestFitnessBeforeLocalSearch
								- DELTA)) : "best fitness before applyLocalSearch() was: " + bestFitnessBeforeLocalSearch
						+ ", now best fitness is " + bestFitnessAfterLocalSearch;
						else
							assert(bestFitnessAfterLocalSearch <= (bestFitnessBeforeLocalSearch
									+ DELTA)) : "best fitness before applyLocalSearch() was: " + bestFitnessBeforeLocalSearch
							+ ", now best fitness is " + bestFitnessAfterLocalSearch;
				}

				double newFitness = getBestFitness();

				if (getFitnessFunction().isMaximizationFunction())
					assert(newFitness >= (bestFitness - DELTA)) : "best fitness was: " + bestFitness
					+ ", now best fitness is " + newFitness;
					else
						assert(newFitness <= (bestFitness + DELTA)) : "best fitness was: " + bestFitness
						+ ", now best fitness is " + newFitness;
						bestFitness = newFitness;

						if (Double.compare(bestFitness, lastBestFitness) == 0) {
							starvationCounter++;
						} else {
							logger.info("reset starvationCounter after " + starvationCounter + " iterations");
							starvationCounter = 0;
							lastBestFitness = bestFitness;

						}

						updateSecondaryCriterion(starvationCounter);

						logger.info("Current iteration: " + currentIteration);
						this.notifyIteration();

						logger.info("Population size: " + population.size());
						logger.info("Best individual has fitness: " + population.get(0).getFitness());
						logger.info("Worst individual has fitness: " + population.get(population.size() - 1).getFitness());					
			}

			TimeController.execute(this::updateBestIndividualFromArchive, "update from archive", 5_000);
			notifySearchFinished();
		 */
		///////////////////////////////////////////////////////////////////////////////////////////////////////			

	}

		
	private double getBestFitness() {
		T bestIndividual = getBestIndividual();
		for (FitnessFunction<T> ff : fitnessFunctions) {
			ff.getFitness(bestIndividual);
		}
		return bestIndividual.getFitness();
	}

	/**
	 * <p>
	 * setReplacementFunction
	 * </p>
	 * 
	 * @param replacement_function
	 *            a {@link org.evosuite.ga.ReplacementFunction} object.
	 */
	public void setReplacementFunction(ReplacementFunction replacement_function) {
		this.replacementFunction = replacement_function;
	}

	/**
	 * <p>
	 * getReplacementFunction
	 * </p>
	 * 
	 * @return a {@link org.evosuite.ga.ReplacementFunction} object.
	 */
	public ReplacementFunction getReplacementFunction() {
		return replacementFunction;
	}

}
