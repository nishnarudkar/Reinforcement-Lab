%% Experiment 8
% Create Discrete Pendulum Environment and Train DQN Agent

clc;
clear;
close all;

%% Step 1: Set random seed
rng(0,"twister");

%% Step 2: Create discrete pendulum environment

env = rlPredefinedEnv("SimplePendulumModel-Discrete");

%% Step 3: Open Simulink model

mdl = "rlSimplePendulumModel";
open_system(mdl);

%% Step 4: Set initial condition

env.ResetFcn = @(in)setVariable(in,"theta0",pi,"Workspace",mdl);

%% Step 5: Get observation and action information

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

disp("Observation Information:");
disp(obsInfo);

disp("Action Information:");
disp(actInfo);

%% Step 6: Create DQN agent

dqnAgent = rlDQNAgent(obsInfo,actInfo);

%% Step 7: Configure DQN agent

dqnAgent.AgentOptions.SampleTime = 0.05;

dqnAgent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-3;

dqnAgent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;

dqnAgent.AgentOptions.ExperienceBufferLength = 1e6;

%% Step 8: Training options

trainOpts = rlTrainingOptions( ...
    MaxEpisodes=500, ...
    MaxStepsPerEpisode=500, ...
    Plots="training-progress");

%% Step 9: Train DQN agent

disp("Starting DQN training...");

trainingStats = train(dqnAgent,env,trainOpts);

%% Step 10: Evaluate trained agent

disp("Evaluating trained agent...");

simOptions = rlSimulationOptions( ...
    MaxSteps=500);

experience = sim(env,dqnAgent,simOptions);

%% Step 11: Calculate total reward

totalReward = sum(experience.Reward);

fprintf("\n====================================\n");
fprintf("Experiment 8 Results\n");
fprintf("====================================\n");
fprintf("Total Evaluation Reward: %.2f\n",totalReward);
fprintf("====================================\n");