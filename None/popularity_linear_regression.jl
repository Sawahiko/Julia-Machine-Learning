using DataFrames, Gadfly, Knet
println("### READING DATA ###")
data = readtable("OnlineNewsPopularity.csv", separator = ',')[:, 3:end]; # first 2 columns have no values
println((:data_shape, size(data)))
println("")
println("# Preparing Data:")
println("* scaling X, log Y, and split data into train and test")
x = convert(Array{Float32}, data[:, 1:end-1]);
y = convert(Array{Float32}, data[end]);
# y = map(Float32, data[:, end])
x = x ./ sum(x, 1); # scale x
y = log.(y); # log y 
splits = round(Int, 0.1 * size(x, 1)); 
shuffled = randperm(size(x, 1));
xtrain, ytrain = [x[shuffled[splits + 1:end], :]', y[shuffled[splits + 1:end]]'];
xtest, ytest = [x[shuffled[1:splits], :]', y[shuffled[1:splits]]'];
println("# Check that both data are of the same distribution")
println(sum(ytrain) / length(ytrain), " , ", sum(ytest) / length(ytest))
println("")
println("# size of train and test data")
println(size(xtrain), size(xtest))
println("")
println("# Define the Model:")
println("* predict, loss, and train functions and initial coeffs and the intercept")
predict(w, x) = w[1] * x .+ w[2]; # linear regression equation ax + b
loss(w, x, y) = mean(abs2, y-predict(w, x)); # mean error
lossgradient = grad(loss); # lossgradient returns dw, the gradient of the loss 
function train(w, data; lr = 0.25) # lr learning rate
    for (x, y) in data
        dw = lossgradient(w, x, y)
        for i in 1:length(w)
           w[i] -= lr * dw[i]
        end     
    end
    return w
end;
w = map(Array{Float32}, Any[0.1 * randn(1, 58), zeros(1, 1)]); # initial coefficients of 58 variables and the intercept
train_loss = [];
test_loss = [];
println("")
println("### FITTING THE MODEL ###")
for i in 1:10
	train(w, [(xtrain, ytrain)])
	append!(train_loss, loss(w, xtrain, ytrain))
	append!(test_loss, loss(w, xtest, ytest))
	println((:epoch, i, :train_loss, train_loss[i], :test_loss, test_loss[i]))
end
println("")
println("# plotting the loss of train and data")
trn = DataFrame(epoch = 1:10, train_loss = train_loss);
tst = DataFrame(epoch = 1:10, test_loss = test_loss);
df = join(trn, tst, on = :epoch) ;
df = melt(df, :epoch);
plt = plot(df, x = :epoch, y = :value, color = :variable, Geom.line);
draw(PNG("loss_plot.png", 6inch, 5inch), plt)
println("# a loss plot has been saved into your directory")

