def testAccuracy(model):
    PCAVector = []
    truth = []
    correctPred = 0
    model.to(device)
    
    for i in range(len(datasetTest)):
        output = model(datasetTest[i][0].unsqueeze(0).to(device))
        output = np.array(output.detach().to('cpu'))
        if datasetTest[i][1] == np.argmax(output[0]):
            correctPred += 1
        PCAVector.append(np.array(activation['avgpool'].to('cpu')).reshape(-1))
        truth.append(datasetTest[i][1])
        print("{:.2f} % ({:d} su {:d}) acc = {:.2f}".format(100*i/len(datasetTest), i, len(datasetTest), 100 * correctPred / (i + 1)), end="\r")
    print("Accuracy of prediction ("+ model.name+ ") "+str(correctPred/len(datasetTest)))
    
    pca = PCA(n_components=2)
    PCAtoplot = pca.fit_transform(np.array(PCAVector))
    PCAtoplot = np.append(PCAtoplot, np.array(truth).reshape(-1, 1), axis=1)
    
    fig = plt.figure(figsize=(10, 7))
    firstLabel = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    for x, y, color in PCAtoplot:
        color = int(color)
        if color == 0:
            if firstLabel[color] == 1:
                plt.plot(x, y, 'bo', label=labelName[color])
                firstLabel[color] = 0
            else:
                plt.plot(x, y, 'bo')
        if color == 1:
            if firstLabel[color] == 1:
                plt.plot(x, y, 'go', label=labelName[color])
                firstLabel[color] = 0
            else:
                plt.plot(x, y, 'go')
        if color == 2:
            if firstLabel[color] == 1:
                plt.plot(x, y, 'ro', label=labelName[color])
                firstLabel[color] = 0
            else:
                plt.plot(x, y, 'ro')
        if color == 3:
            if firstLabel[color] == 1:
                plt.plot(x, y, 'yo', label=labelName[color])
                firstLabel[color] = 0
            else:
                plt.plot(x, y, 'yo')
        if color == 4:
            if firstLabel[color] == 1:
                plt.plot(x, y, 'kd', label=labelName[color])
                firstLabel[color] = 0
            else:
                plt.plot(x, y, 'kd')
        if color == 5:
            if firstLabel[color] == 1:
                plt.plot(x, y, 'ch', label=labelName[color])
                firstLabel[color] = 0
            else:
                plt.plot(x, y, 'ch')
        if color == 6:
            if firstLabel == 1:
                plt.plot(x, y, 'm*', label=labelName[color])
                firstLabel[color] = 0
            else:
                plt.plot(x, y, 'm*')
        if color == 7:
            if firstLabel[color] == 1:
                plt.plot(x, y, 'bs', label=labelName[color])
                firstLabel[color] = 0
            else:
                plt.plot(x, y, 'bs')
            
    plt.ylabel('PC1')
    plt.xlabel('PC2')
    plt.legend(loc="upper right")
    plt.show()
