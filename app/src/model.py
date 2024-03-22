import skimage as ski
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess(img):
    """ Preprocess image for classification
        Steps:
            * Convert to gray scale
            * Normalize
            * Resize to 250x250
            * Invert colors
            (* Threshold to find boxes)

        (Find contours of the boxes)

        (Returns: a tuple with the preprocessed image and an array of the boxes
        Returns: preprocessed image
       """
    if img.shape[-1] == 4:
        img = ski.color.rgba2rgb(img)
    if img.shape[-1] == 3:
        img = ski.color.rgb2gray(img)
    # img = img/255
    img = ski.transform.resize(img, (252, 252))
    img = ski.util.invert(img)
    thresh = ski.filters.threshold_otsu(img, nbins=256)
    img = img > thresh
    # img = ski.filters.sobel(img)
    # # img = ski.filters.gaussian(img, sigma=1)
    # contours = ski.measure.find_contours(img, level=.01)
    # boxes = [c.astype(np.uint16) for c in contours if c.shape[0] > min_shape_size]
    # print(f'Found {len(boxes)} boxes')
    return img

def split_boxes(img, clf):
    predictions = []
    i = 0
    x = np.arange(0, 280, 28)
    y = np.arange(0, 280, 28)
    xx, yy = np.meshgrid(x, y)
    fig, axrows = plt.subplots(9, 9)
    for axrow, row, col in zip(axrows, xx, yy):
        for ax, x, y in zip(axrow, row, col):
            i += 1
            box = img[y:y+28, x:x+28]
            ax.imshow(box, cmap='gray', alpha=.5)
            # ax.text(2, 25, f'{box.mean():.2f}', c='g')
            # ax.text(2, 10, f'{i}', c='b')
            if np.mean(box) > .08:
                pred = clf.predict(box.ravel().reshape(1, -1))
                predictions.append(str(pred[0]))
                ax.text(20, 10, f'{pred[0]}', c='r')
            else:
                predictions.append(' ')
            ax.tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
    su_df = pd.DataFrame(np.array(predictions).reshape(9, 9))
    return fig, su_df

def plot_image(img, boxes):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    for c in boxes:
        ax.plot(c[:, 1], c[:, 0], linewidth=2)
    ax.tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
    return fig

def predict_sudoku(img, clf, boxes, sudoku_true=None):
    """ Given an image of a sudoku, a trained classifier and an array of boxes, try to classify the numbers in the boxes that have numbers in them.

        Returns: a pyplot figure with the image, a pandas DataFrame with the classified values
    """

    predictions = []
    for b in boxes:
        cols = b[:, 1]
        rows = b[:, 0]
        num = img[min(rows)+1:max(rows), min(cols)+1:max(cols)]
        num = ski.transform.resize(num, (28, 28)).ravel()
        if np.mean(num) > .05:
            predictions.append(str(clf.predict(num.reshape(1, -1))[0]))
        else:
            predictions.append(' ')

    su_df = pd.DataFrame(np.array(predictions).reshape(9, 9))
    fig, ax = plt.subplots()
    ax.imshow(ski.util.invert(img), cmap='gray', alpha=.65)
    for box, (i, pred) in zip(boxes, enumerate(predictions)):
        if sudoku_true is not None:
            if predictions[i] == sudoku_true[i]:
                color = 'green'
            else:
                color = 'red'
        else:
            color = 'blue'
        ax.text(np.median(box[:, 1]), np.mean(box[:, 0]), pred, fontdict={'fontsize': 16, 'color': color})
    ax.tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set()
    return fig, su_df

def plot_solution(img, solved):
    """ Plot the solved sudoku

        Returns: a pyplot figure of the sudoku with the solution filled in
    """
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    x = np.arange(0, 280, 28)
    y = np.arange(0, 280, 28)
    xx, yy = np.meshgrid(x, y)
    for (row, col), true_row in zip(zip(xx, yy), solved):
        for x, y, true in zip(row, col, true_row):
            # print(x, y, true)
            num = img[y:y+28, x:x+28]
            if np.mean(num) > .93:
                ax.text(x+10, y+20, true, fontdict={'fontsize': 16, 'color': 'k'})
    ax.tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set()
    return fig

def plot_hint(img, boxes, solved):
    """ Plot the sudoku with one of the solved numbers """
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    for box, (i, true) in zip(boxes, enumerate(solved)):
        cols = box[:, 1]
        rows = box[:, 0]
        num = img[min(rows)+1:max(rows), min(cols)+1:max(cols)]
        if np.mean(num) > 250:
            ax.text(np.mean(box[:, 1]), np.mean(box[:, 0]), true[0], fontdict={'fontsize': 16, 'color': 'k'})
    ax.tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set()
    return fig