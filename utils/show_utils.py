import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

couleur = {"bleu": np.array([0.2, 0.4, 1.0]), "rouge": np.array([1.0, 0.3, 0.3])}

def open_img(path):
    img = Image.open(path).convert("L")  # niveaux de gris
    img_np = np.array(img).astype(np.float32)
    return img_np / 255.0 # Normalisation

def colorize(img, color):
    img_norm = img / 255.0
    return np.stack([img_norm]*3, axis=-1) * 0.3 + couleur[color] * 0.7


def agregate_drawings(X, y, label, fname, outdir="agregated_img", rows=100, cols=50):
    '''
    Agrège les dessins d'une classe donnée en une grande image.
    X : matrice des dessins (n_samples, 784)
    y : vecteur des labels (n_samples,)
    label : classe à afficher
    title : nom de la classe (pour le nom du fichier)
    outdir : répertoire de sauvegarde
    rows, cols : dimensions de la grille d'images à afficher
    Retourne le chemin de l'image sauvegardée.
    '''
    size = rows * cols
    X_class = X[y == label][:size]
    big_img = np.zeros((rows * 28, cols * 28))

    idx = 0
    for r in range(rows):
        for c in range(cols):
            img = X_class[idx].reshape(28, 28)
            big_img[r*28:(r+1)*28, c*28:(c+1)*28] = img
            idx += 1

    os.makedirs(f"img/{outdir}", exist_ok=True)
    filename = os.path.join("img", outdir, f"{fname}_{size}.png")
    plt.imsave(filename, big_img, cmap='gray')

    print(f"Image sauvegardée : {filename}")
    
    return filename


def aggregate_colorize_bad_images(X, bad_indices, label_class, title, outdir="animal/bad_images", cols=10, rows=6):
    ''''
    Agrège les dessins ratés d'une classe donnée en une grande image,
    en colorisant les dessins trop vides en bleu et les dessins trop pleins en rouge
    X : matrice des dessins (n_samples, 784)
    bad_indices : liste de tuples (index, label, too_much)
    label_class : classe à afficher
    title : nom de la classe (pour le nom du fichier)
    outdir : répertoire de sauvegarde
    cols, rows : dimensions de la grille d'images à afficher
    Retourne le chemin de l'image sauvegardée.
    '''
    max_per_class = cols * rows

    bad_imgs = []
    for idx, label, too_much in bad_indices:
        if label != label_class:
            continue
        img = X[idx].reshape(28, 28)
        if too_much == 0 :
            bad_imgs.append(colorize(img, "bleu"))
        elif too_much == 1 :
            bad_imgs.append(colorize(img, "rouge"))
        if len(bad_imgs) >= max_per_class:
            break
        
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    fig.suptitle(title.upper(), fontsize=22, weight='bold')

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx < len(bad_imgs):
                ax.imshow(bad_imgs[idx])
            ax.axis('off')
            idx += 1
         
    os.makedirs(f"img/{outdir}", exist_ok=True)   
    #filename = f"img/{outdir}/{title}_bad.png"    
    filename = os.path.join("img", outdir, f"{title}_bad.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
        
    return filename

def combine_2x2(image_paths, fname="combined", outdir="animal", padding = 20):
    """
    Combine 4 images en une grille 2x2 et ajoute un titre en haut.
    On suppose que toutes les images ont la même taille
    image_paths : liste de 4 chemins d'images
    """

    imgs = [Image.open(p) for p in image_paths]
    w, h = imgs[0].size

    # Nouvelle image blanche
    combined = Image.new("RGB", (2 * w + padding, 2 * h + padding), color=(255, 255, 255))
    draw = ImageDraw.Draw(combined)

    # Placement des 4 images
    combined.paste(imgs[0], (0, 0))
    combined.paste(imgs[1], (w + padding, 0  ))
    combined.paste(imgs[2], (0, h + padding))
    combined.paste(imgs[3], (w + padding, h + padding))

    filename = os.path.join("img", outdir, f"{fname}.png")
    combined.save(filename)
    return filename

def add_title(image_path, title, fontsize=48, padding=60):
    """
    Ajoute un titre centré au-dessus d'une image et retourne une nouvelle image.
    
    image_path : chemin de l'image à modifier
    title : texte du titre
    fontsize : taille du texte
    padding : espace vertical entre le titre et l'image
    """
    image = Image.open(image_path)

    # Charger la police arial
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except:
        font = ImageFont.load_default()

    w, h = image.size

    # Mesure du texte
    dummy = ImageDraw.Draw(image)
    bbox = dummy.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    new_h = h + text_h + padding

    new_img = Image.new("RGB", (w, new_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(new_img)

    draw.text(((w - text_w) // 2, padding // 2), title, fill=(0, 0, 0), font=font)

    new_img.paste(image, (0, text_h + padding))

    new_img.save(image_path)
    return new_img


def add_caption(image_path, caption, fontsize=32, padding=20):
    """
    Ajoute une légende sous une image et retourne une nouvelle image.
    """
    image = Image.open(image_path)
    
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except:
        font = ImageFont.load_default()

    w, h = image.size
    dummy = ImageDraw.Draw(image)
    bbox = dummy.textbbox((0, 0), caption, font=font) 
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    new_h = h + text_h + padding
    new_img = Image.new("RGB", (w, new_h), color=(255, 255, 255))

    new_img.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_img)
    draw.text(((w - text_w) // 2, h + padding // 2), caption, fill=(0, 0, 0), font=font)

    new_img.save(image_path)
    return image_path


def count_bad_drawings(bad_indices, nb_classes):
    ''''
    Compte le nombre de dessins ratés pour chaque classe, 
    séparément pour les dessins trop vides et trop pleins.
    bad_indices : liste de tuples (index, label, too_much)
    nb_classes : nombre total de classes
    Retourne deux listes : counts_low et counts_high,
    de taille (nb_classes + 1), avec un total à la fin de chaque liste.
    '''
    counts_low = [0] * nb_classes
    counts_high = [0] * nb_classes

    for idx, label, err in bad_indices:
        if err == 0:
            counts_low[label] += 1
        else:
            counts_high[label] += 1

    total_low = sum(counts_low)
    total_high = sum(counts_high)

    return counts_low + [total_low], counts_high + [total_high]

def plot_bad_drawing_bars(bad_indices, animaux):
    counts_low, counts_high = count_bad_drawings(bad_indices, len(animaux))

    labels = animaux + ["Total"]
    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))

    plt.bar(x, counts_low, color=(0.2, 0.4, 1.0), label="Trop vide")
    plt.bar(x, counts_high, bottom=counts_low, color=(1.0, 0.3, 0.3), label="Trop plein")

    for i in range(len(labels)):
        plt.text(x[i], counts_low[i] / 2, str(counts_low[i]), color='white', 
                 ha='center', va='center', fontsize=10, fontweight='bold')

        plt.text(x[i], counts_low[i] + counts_high[i] / 2, str(counts_high[i]), color='beige', 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        plt.text(x[i], counts_low[i] + counts_high[i] + 30 / 2, 
                str(counts_low[i] + counts_high[i]), color='black', 
                ha='center', va='center', fontsize=10, fontweight='bold')

    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel("Nombre de dessins ratés")
    plt.title("Répartition des dessins ratés par classe", fontsize=18, weight='bold')
    plt.legend()

    plt.tight_layout()
    plt.show()

def save_predictions_grid(X, y_true, model, labels, title="Prédictions du modèle", 
                          save_name=None, save_dir="./img/animal", rows=10, cols=8):
    n = rows * cols
    indices = np.random.choice(len(X), n, replace=False)

    fig = plt.figure(figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(title, fontsize=18, weight='bold')

    for i, idx in enumerate(indices):
        x = X[idx]
        img = x.reshape(28, 28) if x.shape == (784,) else x

        # Toujours transformer en batch
        x2 = np.atleast_2d(x)

        # Probabilités selon le type de modèle
        if hasattr(model, "forward"):  
            P = model.forward(x2.T, training=False).flatten()
        elif hasattr(model, "predict_proba"):   # (ex: sklearn)
            P = model.predict_proba(x2)[0]
        else:
            raise ValueError("Modèle incompatible : besoin de forward() ou predict_proba().")

        pred = int(np.argmax(P))
        prob = float(P[pred])

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img, cmap='gray')
        ax.set_title(
            f"{labels[pred]} ({prob*100:.1f}%)\nVrai : {labels[y_true[idx]]}",
            fontsize=7
        )
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_name is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{save_name}.png")
        plt.savefig(path, dpi=150)
        print(f"Image sauvegardée dans : {path}")

    plt.show()
    
def save_predictions_grid(X, y_true, preds, probas, labels, title="Prédictions du modèle",
                          save_name=None, save_dir="./img/animal", rows=10, cols=8):

    n = rows * cols
    indices = np.random.choice(len(X), n, replace=False)

    fig = plt.figure(figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(title, fontsize=18, weight='bold')

    for i, idx in enumerate(indices):
        x = X[idx]
        img = x.reshape(28, 28) if x.shape == (784,) else x

        pred = preds[idx]
        true = y_true[idx]
        prob = probas[pred] if probas.ndim == 1 else probas[pred, idx]   # probabilité de la classe prédite

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img, cmap='gray')
        ax.set_title(
            f"{labels[pred]} ({prob*100:.1f}%)\nVrai : {labels[true]}",
            fontsize=7
        )
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_name is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{save_name}.png")
        plt.savefig(path, dpi=150)
        print(f"Image sauvegardée dans : {path}")


def save_merge_img(img, title, save_name=None, save_dir="./img"):
    n = len(img)
    if len(title) != n:
        raise ValueError("Le nombre de titres doit correspondre au nombre d'images.")

    plt.figure(figsize=(4*n,4))
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.title(title[i])
        plt.imshow(img[i], cmap="gray")
        plt.axis("off")

    if save_name is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{save_name}.png")
        plt.savefig(path, bbox_inches='tight')
        print(f"Image sauvegardée dans : {path}")
        
    plt.close()
    
# =============== Affichage de l'architecture des modèles ===============
    
def print_MLP_architecture(model_MLP):
    print("Couches = ", [w.shape[1] for w in model_MLP.W] + [model_MLP.W[-1].shape[0]])
    
def print_CNN_architecture(model_CNN):
    for i, layer in enumerate(model_CNN.layers):
        txt = f"Layer {i}: {type(layer).__name__}"
        if hasattr(layer, "C_out") : # ConvLayer
            txt += f" [{layer.C_out} filtres]  (k={layer.k}, stride={layer.stride}, pad={layer.pad})"
        elif hasattr(layer, "L"): # MLP
            txt += f", Couches = {[w.shape[1] for w in layer.W] + [layer.W[-1].shape[0]]}"
        print(txt)

def print_CNNTorch_architecture(model_TorchCNN):
    i = 0
    for layer in model_TorchCNN.features:
        txt = f"Layer {i} : {type(layer).__name__}"
        if hasattr(layer, 'out_channels'):  # Conv2d layer
            txt += f" [{layer.out_channels} filtres]  (k={layer.kernel_size[0]}, stride={layer.stride[0]}, pad={layer.padding[0]})"
        print(txt)
        i += 1
    print(f"Layer {i} : Flatten")
    txt = f"Layer {i+1} : MLP, Couches = ["
    for layer in model_TorchCNN.classifier:  
        if hasattr(layer, 'out_features'):  # Linear layer
            txt += f"{layer.out_features}, "
    print(txt + f"{model_TorchCNN.classifier[-1].out_features}]")

def show_img(image_path, height=16, width=12):
    img = Image.open(image_path)
    plt.figure(figsize=(height, width))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

