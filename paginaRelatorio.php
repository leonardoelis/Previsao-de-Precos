<?php
    session_start();

    if (empty($_SESSION['ra']) and  empty($_SESSION['senha'])) {
        header('Location: login.php');
    }
    
    require_once 'banco.php';
    $banco = new Banco();
?>

<html>
    <head>
        <meta charset="utf-8">
        <title>Pesquisa CPA</title>
    </head>
    <body>
        <form action="paginaRelatorio_resp.php" method="post">
            <input type="text" name="bloco">
            <input type="submit" value="Gerar Relatório">
        </form>
    </body>
</html>
